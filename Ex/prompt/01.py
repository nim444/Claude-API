import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-6"
message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What is quantum computing? answer in one sentence.",
        }
    ],
)

print(message)

# Message(
# id='msg_015NzjANKFrTBeLtK3tWkp4Y',
# container=None, content=[TextBlock(citations=None,
# text='Quantum computing is a type of computing that uses quantum mechanical phenomena, such as superposition and entanglement, to process information in ways that can solve certain complex problems much faster than classical computers.',
# type='text')],
# model='claude-sonnet-4-6',
# role='assistant',
# stop_reason='end_turn',
# stop_sequence=None,
# type='message',
# usage=Usage(cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0),
# cache_creation_input_tokens=0, cache_read_input_tokens=0, inference_geo='global', input_tokens=17,
# output_tokens=43, server_tool_use=None, service_tier='standard'))
