import re

from typing import Tuple

from utils import CompletionGPT, ChatGPT, DIALOGUES, ChatDeepSeek, CompletionDeepSeek

import requests
import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import List, Dict, Any, Union, Optional

# import TimeoutException
from requests.exceptions import Timeout, ConnectionError
from fastchat.model.model_adapter import get_conversation_template

import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import ipdb

class Agent:
    def __init__(self, **configs) -> None:
        self.name = configs.pop("name", None)
        self.src = configs.pop("src", None)
        pass

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError

class Prompter:
    @staticmethod
    def get_prompter(prompter_name: Union[str, None]):
        # check if prompter_name is a method and its variable
        if not prompter_name:
            return None
        if hasattr(Prompter, prompter_name) and callable(getattr(Prompter, prompter_name)):
            return getattr(Prompter, prompter_name)
    
    @staticmethod
    def claude(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "Human",
            "agent": "Assistant",
        }
        for item in messages:
            prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
        prompt += "Assistant:"
        return {"prompt": prompt}

    @staticmethod
    def openchat_v3_1(messages: List[Dict[str, str]]):
        prompt = "Assistant is GPT4<|end_of_turn|>"
        role_dict = {
            "user": "User: {content}<|end_of_turn|>",
            "agent": "Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "Assistant:"
        return {"prompt": prompt}
    
    @staticmethod
    def openchat_v3_2(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "GPT4 User: {content}<|end_of_turn|>",
            "agent": "GPT4 Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "GPT4 Assistant:"
        return {"prompt": prompt}


class FastChatAgent(Agent):
    def __init__(self, model_name, controller_address=None, worker_address=None, temperature=0, max_new_tokens=32, top_p=0, prompter=None, args=None, **kwargs) -> None:
        if controller_address is None and worker_address is None:
            raise ValueError("Either controller_address or worker_address must be specified.")
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            worker_addr = controller_addr
        if worker_addr == "":
            return
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args
        }
        if self.prompter:
            prompt = self.prompter(history)
            gen_params.update(prompt)
        else:
            conv = get_conversation_template(self.model_name)

            for history_item in history:
                role = history_item["role"]
                content = history_item["content"]
                if role == "user" or role == "system":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            if history[-1]["role"] == conv.roles[1]:
                print("finish agent's response")
            else:
                print("generate a new response")
                conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            gen_params.update({
                "prompt": prompt,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
            })
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        text = json.loads(line)["text"]
                return text
            # if timeout or connection error, retry
            except Timeout: 
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
                # ipdb.set_trace()
            time.sleep(60)
        else:
            raise Exception("Timeout after 3 retries.")


class HfChatAgent(Agent):
    """
    An Agent that uses a locally loaded Hugging Face model for inference,
    avoiding FastChat/controller/worker servers, and supports batched inference.
    """
    def __init__(
        self,
        model_name: str = "gpt2",
        temperature: float = 0.0,
        max_new_tokens: int = 32,
        top_p: float = 0.0,
        prompter=None,                   # Pass a callable that formats prompts, if desired
        device: Optional[str] = None,    # "cuda" or "cpu"
        args: Optional[Dict] = None,
        batch_size: int = 1,            # <-- Default batch size
        **kwargs
    ) -> None:
        """
        :param model_name: Hugging Face model name.
        :param temperature: Sampling temperature.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param top_p: Top-p (nucleus) sampling.
        :param prompter: Custom function to build/format the prompt from history.
        :param device: "cuda" or "cpu". Defaults to GPU if available.
        :param args: Extra generation arguments.
        :param batch_size: Default batch size for batched inference.
        :param kwargs: Passed to the base Agent class if needed.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        self.batch_size = batch_size

        # Determine default device if not explicitly given
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model {model_name} on device {device}")

        # Load tokenizer and model locally
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Create a text-generation pipeline
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=1 if device == "cuda" else -1  # pipeline device argument
        )

    def _build_prompt(self, history: List[dict]) -> str:
        """
        Internal helper to build a single textual prompt from a single conversation history.
        """
        if self.prompter:
            # If a custom prompter was passed in, use it
            return self.prompter(history)

        # Otherwise, a simple concatenation approach:
        prompt = ""
        for turn in history:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "agent":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant: "
        return prompt

    def inference(self, history: List[dict]) -> str:
        """
        Generates a response given the conversation history.
        
        :param history: List of dicts, each with {"role": "user"|"agent"|"system", "content": "..."}.
        :return: Generated text from the local model.
        """
        # 1. Build or retrieve the prompt.
        if self.prompter:
            # If a custom prompter was passed in, use it to create the prompt.
            prompt = self.prompter(history)
            # Ensure it's a string or a dict that your pipeline can handle
        else:
            # Simple concatenation approach:
            prompt = ""
            for turn in history:
                role = turn["role"]
                content = turn["content"]
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "agent":
                    prompt += f"Assistant: {content}\n"
                elif role == "system":
                    prompt += f"System: {content}\n"
                else:
                    prompt += f"{role.capitalize()}: {content}\n"
            # Typically end with an Assistant prompt:
            prompt += "Assistant: "

        # 2. Generate text locally with the pipeline.
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=(self.temperature > 0),  # do_sample only if temperature > 0
            **self.args
        )

        # 3. Extract generated text
        generated_text = outputs[0]["generated_text"]

        # 4. Optionally, remove the original prompt part from the output (if desired)
        #    This depends on whether you want the full text or just the newly generated content.
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt) :].strip()
        else:
            response = generated_text

        return response

    def batched_inference(
        self,
        histories: Union[List[dict], List[List[dict]]],
        batch_size: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Generates responses given one or multiple conversation histories.
        
        :param histories: Either a single conversation (List[dict]) or a list of 
                          conversations (List[List[dict]]).
        :param batch_size: If provided, overrides the default batch size.
        :return: A single string (for single history) or a list of strings (for batched input).
        """
        if not histories:
            return ""

        # Check if we have a single conversation or a batch
        is_batch = isinstance(histories[0], list)
        
        # Convert a single conversation to a batch of size 1
        if not is_batch:
            histories = [histories]  # Wrap in list so we can handle uniformly

        # Build prompts for each conversation
        prompts = [self._build_prompt(h) for h in histories]

        # Use the requested batch size, or fall back to the default
        if batch_size is None:
            batch_size = self.batch_size

        # Generate with pipeline (handles batches automatically if given a list of strings)
        outputs = self.generator(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=(self.temperature > 0),
            batch_size=batch_size,      # <-- The key for controlling batch size
            **self.args
        )

        # 'outputs' will be a list of lists (one sub-list per input prompt).
        responses = []
        for i, out in enumerate(outputs):
            generated_text = out[0]["generated_text"]
            # Strip out the prompt part if desired
            if generated_text.startswith(prompts[i]):
                response = generated_text[len(prompts[i]):].strip()
            else:
                response = generated_text
            responses.append(response)

        # If originally not a batch, return the single string instead of a list
        if not is_batch:
            return responses[0]
        else:
            return responses
    

class BasePolicy:
    def __init__(self):
        pass

    def forward(query, observation, available_actions):
        raise NotImplementedError
    
    def init_dialogue(self, role, env):
        if env in DIALOGUES:
            dialogue = copy.deepcopy(DIALOGUES[env])
            if len(dialogue) >= 2:
                dialogue[1]["role"] = role
        else:
            dialogue = []
        return dialogue


class HfChatPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "", response_limit: int = 1000, device: str = "cpu", batch_size: int = 1):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit
        self.dialogues = {}
        self.agent = HfChatAgent(model_name=model, temperature=1.0, device=device, max_new_tokens=response_limit, top_p=1.0, batch_size=batch_size, prompter=None, args=None, name = "HfChatAgent")

    def reset(self, env):
        self.dialogue = self.init_dialogue("agent", env)
        
    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]
        
        actions = []
        for i in range(num_of_samples):
            raw_actions = self.agent.inference(self.dialogue)
            action = raw_actions[0] if isinstance(raw_actions, list) else raw_actions
            # if action not in actions:
            actions.append(action)
        return actions
        
    def batched_forward(self, num_of_samples, batch_indices, batch_size) -> Union[str, List[str]]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]
        
        batch_actions = {}
        action_list = [[] for _ in range(batch_size)]
        
        batch_dialoues = []
        for index in batch_indices:
            batch_dialoues.append(self.dialogues[index])
            
        for i in range(num_of_samples):
            batch_raw_actions = self.agent.batched_inference(batch_dialoues, batch_size)
            for j in batch_size:
                action = batch_raw_actions[j][0] if isinstance(batch_raw_actions[j], list) else batch_raw_actions[j]
                # if action not in actions:
                action_list[j].append(action)
                
        return action_list
        
    
class ChatGPTPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "gpt-4-turbo-preview", response_limit: int = 1000):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit
        print(f"Teacher Model is {self.model}")

    def reset(self, env):
        self.dialogue = self.init_dialogue("assistant", env)

    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        actions = []

        raw_actions = ChatGPT(self.dialogue, model=self.model, num_samples=num_of_samples)
        # for action in raw_actions:
        #     if action not in actions:
        #         actions.append(action)
        return raw_actions
    
    
class DeepSeekPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "deepseek-chat", response_limit: int = 1000):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit
        print(f"Teacher Model is {self.model}")

    def reset(self, env):
        self.dialogue = self.init_dialogue("assistant", env)

    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from DeepSeek
        actions = []

        raw_actions = ChatDeepSeek(self.dialogue, model=self.model, num_samples=num_of_samples)
        # for action in raw_actions:
        #     if action not in actions:
        #         actions.append(action)
        return raw_actions


class FastChatPolicy(BasePolicy):
    def __init__(self, dialogue_limit: int = None, model: str = "", response_limit: int = 1000, controller_address=21002):
        super().__init__()
        self.dialogue_limit = dialogue_limit
        self.model = model
        self.response_limit = response_limit

        self.agent = FastChatAgent(model_name=model, controller_address=f"http://localhost:{controller_address}", worker_address=None, temperature=1.0, max_new_tokens=response_limit, top_p=1.0, prompter=None, args=None, name="FastChatAgent")
        
    def reset(self, env):
        self.dialogue = self.init_dialogue("agent", env)

    def forward(self, num_of_samples) -> Tuple[str, bool]:
        # Only keep {self.dialogue_limit} most recent messages
        if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
            self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        actions = []
        for i in range(num_of_samples):
            raw_actions = self.agent.inference(self.dialogue)
            action = raw_actions[0] if isinstance(raw_actions, list) else raw_actions
            # if action not in actions:
            actions.append(action)
        return actions