import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = "claude-haiku-4-5"


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    return response.content[0].text


def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing a task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
  {
    "task": "Create a JSON configuration for an AWS Lambda function that sets up a basic Python runtime with a memory allocation of 512MB and a timeout of 10 seconds",
    "format": "json",
    "solution_criteria": "Must include runtime, memory size, timeout, and basic structure for AWS Lambda configuration"
  },
  {
    "task": "Write a Python function to validate AWS S3 bucket names according to AWS naming rules",
    "format": "python",
    "solution_criteria": "Must validate bucket name length (3-63 chars), allowed characters (lowercase, numbers, hyphens), and start/end rules"
  },
  {
    "task": "Write a regex pattern that matches valid AWS EC2 instance IDs",
    "format": "regex",
    "solution_criteria": "Must match the pattern: i- followed by 8 or 17 hexadecimal characters"
  }
]
```

Requirements:
* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code
* Vary the formats: include one task each for "python", "json", and "regex"
* Each task should be AWS-specific and concise
* Include clear, specific solution_criteria that describe what makes a good solution

Please generate 3 objects.
"""
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)


if __name__ == "__main__":
    print("Generating evaluation dataset...")
    dataset = generate_dataset()
    print(dataset)

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} items to dataset.json")
