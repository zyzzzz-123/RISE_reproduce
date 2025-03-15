import config
import openai
import os
import re
from time import sleep
import ipdb

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Set OpenAPI key from environment or config file
api_key = os.environ.get("OPENAI_API_KEY")
# if (api_key is None or api_key == "") and os.path.isfile(os.path.join(os.getcwd(), "keys.cfg")):
# ipdb.set_trace()
if os.path.isfile(os.path.join(os.getcwd(), "keys_deepseek.cfg")):
    cfg = config.Config('keys_deepseek.cfg')
    api_key = cfg.get("DEEPSEEK_API_KEY")
openai.api_key = api_key

openai.api_base = "https://api.deepseek.com/v1"

# Define retry decorator to handle OpenAI API timeouts
@retry(wait=wait_random_exponential(min=20, max=100), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


# Define retry decorator to handle OpenAI API timeouts
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    kwargs["request_timeout"] = 20
    # ipdb.set_trace()
    response = openai.ChatCompletion.create(**kwargs)
    return response

# Define GPT-3 completion function
def CompletionDeepSeek(phrase, model="deepseek-chat", num_samples=1):
    sleep(2)
    response = completion_with_backoff(
        model=model,
        prompt=phrase.strip(),
        temperature=0,
        top_p=1,
        max_tokens=512,
        n=num_samples
    )
    candidates = []
    for candidate in response.choices:
        z = candidate.text
        pred = re.sub("\n"," ", z)
        candidates.append(pred.strip())
    return candidates

# Define GPT-3.5+ chat function
def ChatDeepSeek(messages, model="deepseek-chat", num_samples=1):
    response = chat_with_backoff(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=1,
        max_tokens=512,
        n=num_samples
    )
    candidates = []
    for candidate in response.choices:
        z = candidate.message.content
        # pred = re.sub("\n"," ", z)
        # candidates.append(pred.strip())
        candidates.append(z)
    return candidates
    
if __name__ == "__main__":
    pass
