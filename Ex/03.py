import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-6"


def add_user_message(message, text):
    user_message = {
        "role": "user",
        "content": text,
    }
    message.append(user_message)


def add_assistant_message(message, text):
    assistant_message = {
        "role": "assistant",
        "content": text,
    }
    message.append(assistant_message)


def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text


messages = []

while True:
    user_input = input("> ")
    print(f"User: {user_input}")
    add_user_message(messages, user_input)
    answer = chat(messages)
    add_assistant_message(messages, answer)
    print("-" * 20)
    print(answer)
