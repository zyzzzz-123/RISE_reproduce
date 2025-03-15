import json
import os
import re
import torch
import argparse, json, os, re
from tqdm import tqdm
from typing import Dict, List
import sys
from rich import print

from policies import ChatGPTPolicy, FastChatPolicy, DeepSeekPolicy, HfChatPolicy

from environments.gsm8k import GSM8KEnv
from environments.math import MATHEnv
import ipdb

parser = argparse.ArgumentParser(description='N-turn evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, default="/Users/yxqu/Desktop/CMU/Research/Self-Imrovement/data/gsm8k/demo.jsonl", help='path to dataset to evaluate on')
parser.add_argument('--dialogue_limit', type=int, default=20, help='maximum number of turns in the policy\'s dialogue to keep')
parser.add_argument('--env', choices=['gsm8k', 'math'], default="gsm8k", help='environment to run eval on')
parser.add_argument('--log_dir', type=str, default="./", help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, default=5, help='max number of interaction turns')
parser.add_argument('--models', nargs='+', type=str, help="models to use for policy")
parser.add_argument('--controller_address', type=str, default="21002", help="model to use for policy")
parser.add_argument('--num_of_samples', nargs='+', type=int, help='number of actions generated each turn')
parser.add_argument('--verbose', action='store_true', help="print out logs")
parser.add_argument('--debug', action='store_true', help="debugging mode with less data")
parser.add_argument('--model_suffix', type=str, default="", help="specify different model output")
parser.add_argument('--batch_size', type=int, default=1, help="batch size for evaluation")
parser.add_argument('--without_controller', action='store_true', help="use model without controller")
args = parser.parse_args()
print(args)

def detect_duplicates(text, threshold):
    # Convert the text to lowercase and split it into words
    words = text.lower().split()

    # Create a dictionary to store word counts
    word_counts = {}

    # Count the occurrences of each word
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Check if any word count exceeds the threshold
    for word, count in word_counts.items():
        if count > threshold:
            return True
        if len(word) > 300:
            return True
    else:
        return False


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
        
class ExperimentWrapper():
    def __init__(self, args):
        self.args = args

        # Set environment (No logging for env)
        self.env = None
        if args.env == 'gsm8k':
            self.env = GSM8KEnv(args.data_path)
        elif args.env == 'math':
            self.env = MATHEnv(args.data_path)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        model_suffix = '_' + args.model_suffix if args.model_suffix else ''
        log_file_name = f"{args.max_turns}_turns{model_suffix}.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as fp:
                self.log_data = json.load(fp)

    def init_policy(self, model):
        if "gpt" in model:
            self.policy = ChatGPTPolicy(dialogue_limit=args.dialogue_limit, model=model)
            self.role = "assistant"
        if "deepseek-chat" in model:
            self.policy = DeepSeekPolicy(dialogue_limit=args.dialogue_limit, model=model)
            self.role = "assistant"
        elif self.args.without_controller:
            self.policy = HfChatPolicy(dialogue_limit=args.dialogue_limit, model=model, device="cuda", batch_size=args.batch_size)
            self.role = "agent"
        else:
            self.policy = FastChatPolicy(dialogue_limit=args.dialogue_limit, model=model, controller_address=args.controller_address)
            self.role = "agent"
        return self.policy, self.role

    def construct_policy_dialogue(self, observations, actions, model, turn):
        # self.init_policy(model)
        self.policy = self.policies[turn]
        self.role = self.roles[turn]
        
        self.policy.reset(self.args.env)
        for i in range(len(actions)):
            action = actions[i]
            observation = observations[i]
            self.policy.dialogue.append({"role": "user", "content": observation})
            self.policy.dialogue.append({"role": self.role, "content": action})
            
        self.policy.dialogue.append({"role": "user", "content": observations[-1]})
        
    def construct_batched_policy_dialogue(self, batch_indices, batch_histories, batch_init_observation, batch_query, model, turn, batch_size):
        # self.init_policy(model)
        self.policy = self.policies[turn]
        self.role = self.roles[turn]
        
        self.policy.reset(self.args.env)
        
        for batch_idx in range(batch_size):
            if self.policy.dialogues[batch_indices[i]] is None:
                self.policy.dialogues[batch_indices[i]] = []
                
            dialogue = []
            actions = batch_histories[batch_idx]["best_actions"]
            observations = batch_init_observation[batch_idx] + batch_histories[batch_idx]["best_observations"]
            for i in range(len(actions)):
                action = actions[i]
                observation = observations[i]
                dialogue.append({"role": "user", "content": observation})
                dialogue.append({"role": self.role, "content": action})
            
            dialogue.append({"role": "user", "content": observations[-1]})
            
            self.policy.dialogues[batch_indices[i]].append(dialogue)

    def run_expr(self):
        if self.args.debug:
            self.env.data = self.env.data[:10]
        try:
            self.policies = []
            self.roles = []
            for i in range(self.args.max_turns):
                # initiliaze policies with roles
                policy, role = self.init_policy(self.args.models[i])
                self.policies.append(policy)
                self.roles.append(role)
                
            for idx in tqdm(range(0,len(self.env.data)), disable=self.args.verbose):
                observation, reward, valid_action = None, None, None
                turn_history = {"best_actions": [], "actions": {}, "best_observations": [], "observations": {}, "best_rewards": [], "rewards": {}}
                
                init_observation = self.env.reset(idx)
                query = self.env.query
                if str(idx) in self.log_data:
                    if query == self.log_data[str(idx)]["query"]:
                        print(f"Skipping index {idx} as it has already been processed")
                        continue
                if self.args.verbose:
                    print(f'------\nQuery {idx}: {query}')    

                for turn in range(self.args.max_turns):
                    # ipdb.set_trace()
                    self.construct_policy_dialogue([init_observation] + turn_history["best_observations"], turn_history["best_actions"], args.models[turn], turn)
                    actions = []
                    try:
                        while len(actions) == 0:
                            actions = self.policy.forward(args.num_of_samples[turn])
                            for action in actions:
                                if detect_duplicates(action, 150):
                                    print(f"[WARNING] Index {idx}: Duplicate detected in action: {action}")
                                    actions.remove(action)
                                
                    except (ValueError, TypeError) as e:
                        print(f"[ERROR] Index {idx}: {e}")
                        # Logging
                        turn_history["actions"][turn] = ["blocked"]
                        turn_history["rewards"][turn] = [None]
                        break

                    turn_history["actions"][turn] = actions
                    turn_history["rewards"][turn] = []
                    turn_history["observations"][turn] = []

                    for action in actions:
                        reward, error_message, success = self.env.step(action)
                        observation = self.env.format_output(error_message, success, reward)
                        turn_history["rewards"][turn].append(reward)
                        turn_history["observations"][turn].append(observation)
                    
                    max_reward_idx = turn_history["rewards"][turn].index(max(turn_history["rewards"][turn]))
                    
                    best_action = turn_history["actions"][turn][max_reward_idx]
                    best_reward = turn_history["rewards"][turn][max_reward_idx]
                    best_observation = turn_history["observations"][turn][max_reward_idx]

                    turn_history["best_actions"].append(best_action)
                    turn_history["best_rewards"].append(best_reward)
                    turn_history["best_observations"].append(best_observation)

                    if self.args.verbose:
                        print(f"- Turn {turn}")
                        print(f"-- Best Action: {best_action}")
                        print(f"-- Best Observation: {best_observation}")
                        print(f"-- Best Reward: {best_reward}")
                        
                    if turn_history["best_rewards"][-1] == 1.0:
                        break

                max_reward = max(turn_history["best_rewards"])
                answer = self.env.answer
                log_episode = {
                    "task_id": idx,
                    "query": query,
                    "answer": answer,
                    "max_reward": max_reward,
                    "turn_history": turn_history,
                    "dialogue": self.policy.dialogue
                }
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn+1}")
            print(f"log path: {self.log_path}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()

    def run_batched_exp(self): 
        if self.args.debug:
            self.env.data = self.env.data[:10]
        try:
            self.policies = []
            self.roles = []
            for i in range(self.args.max_turns):
                # initiliaze policies with roles
                policy, role = self.init_policy(self.args.models[i])
                self.policies.append(policy)
                self.roles.append(role)
                
            temp_log_data = {}
            turn_histories = {}
            for turn in range(self.args.max_turns):
                self.policy = self.policies[turn]
                self.role = self.roles[turn]
                
                batch_histories = []
                for idx in tqdm(range(0,len(self.env.data)), disable=self.args.verbose):
                    if args.num_of_samples[turn] == 1:
                        batch_histories = []
                        batch_init_observation = []
                        batch_query = []
                        batch_indices = []
                        for batch_idx in range(self.args.batch_size):
                            if turn == 0:
                                observation, reward, valid_action = None, None, None
                                turn_history = {"best_actions": [], "actions": {}, "best_observations": [], "observations": {}, "best_rewards": [], "rewards": {}}
                                turn_histories[idx + batch_idx] = turn_history
                                
                                batch_histories.append(turn_history)
                                batch_init_observation.append(self.env.reset(idx + batch_idx))
                                query = self.env.query
                                batch_query.append(query)
                                batch_indices.append(idx + batch_idx)
                                
                                if str(idx + batch_idx) in self.log_data:
                                    if query == self.log_data[str(idx + batch_idx)]["query"]:
                                        print(f"Skipping index {idx + batch_idx} as it has already been processed")
                                        continue
                                
                                if self.args.verbose:
                                    print(f'------\nQuery {idx + batch_idx}: {query}')    
                            
                            else: 
                                turn_history = turn_histories[idx + batch_idx]
                                batch_histories.append(turn_history)
                                batch_init_observation.append(self.env.reset(idx + batch_idx))
                                batch_query.append(self.env.query)
                                batch_indices.append(idx + batch_idx)

                        self.construct_batched_policy_dialogue(batch_indices, batch_histories, batch_init_observation, batch_query, args.models[turn], turn, args.batch_size)
                        actions = []
                        try:
                            # while len(actions) == 0:
                            batch_actions = self.policy.batched_forward(args.num_of_samples[turn], batch_indices, args.batch_size)
                                # for action in actions:
                                #     if detect_duplicates(action, 150):
                                #         print(f"[WARNING] Index {idx + batch_idx}: Duplicate detected in action: {action}")
                                #         actions.remove(action)
                                    
                        except (ValueError, TypeError) as e:
                            print(f"[ERROR] Index {idx + batch_idx}: {e}")
                            # Logging
                            turn_histories[idx + batch_idx]["actions"][turn] = ["blocked"]
                            turn_histories[idx + batch_idx]["rewards"][turn] = [None]
                            break
                        
                        for batch_idx in range(self.args.batch_size):
                            actions = batch_actions[batch_idx]
                            turn_histories[idx + batch_idx]["actions"][turn] = actions
                            turn_histories[idx + batch_idx]["rewards"][turn] = []
                            turn_histories[idx + batch_idx]["observations"][turn] = []

                            for action in actions:
                                reward, error_message, success = self.env.step(action)
                                observation = self.env.format_output(error_message, success, reward)
                                turn_histories[idx + batch_idx]["rewards"][turn].append(reward)
                                turn_histories[idx + batch_idx]["observations"][turn].append(observation)
                            
                            max_reward_idx = turn_histories[idx + batch_idx]["rewards"][turn].index(max(turn_histories[idx + batch_idx]["rewards"][turn]))
                            
                            best_action = turn_histories[idx + batch_idx]["actions"][turn][max_reward_idx]
                            best_reward = turn_histories[idx + batch_idx]["rewards"][turn][max_reward_idx]
                            best_observation = turn_histories[idx + batch_idx]["observations"][turn][max_reward_idx]

                            turn_histories[idx + batch_idx]["best_actions"].append(best_action)
                            turn_histories[idx + batch_idx]["best_rewards"].append(best_reward)
                            turn_histories[idx + batch_idx]["best_observations"].append(best_observation)

                            if self.args.verbose:
                                print(f"- Turn {turn}")
                                print(f"-- Best Action: {best_action}")
                                print(f"-- Best Observation: {best_observation}")
                                print(f"-- Best Reward: {best_reward}")
                                
                            if turn_histories[idx + batch_idx]["best_rewards"][-1] == 1.0:
                                break
                
                            if turn == self.args.max_turns - 1:
                                for idx in range(0,len(self.env.data)):
                                    max_reward = max(turn_histories[idx + batch_idx]["best_rewards"])
                                    answer = self.env.answer
                                    log_episode = {
                                        "task_id": idx,
                                        "query": query,
                                        "answer": answer,
                                        "max_reward": max_reward,
                                        "turn_history": turn_history,
                                        "dialogue": self.policy.dialogue
                                    }
                                    self.log_data[idx] = log_episode

            if self.args.verbose:
                print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn+1}")
            print(f"log path: {self.log_path}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()
        

if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()



