"""
Client examples for the ant-ling Chat API.

Shows usage with:
  1. OpenAI Python SDK
  2. Anthropic Python SDK
  3. Raw HTTP (httpx)

Prerequisites:
  pip install openai anthropic httpx
"""

import os

# ── Configuration ──────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "sk-no-key-required")
MODEL = "Ring-2.6-1T"


def openai_non_streaming():
    """OpenAI SDK — non-streaming."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{API_BASE}/v1", api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant named lingg."},
            {"role": "user", "content": "Hello! What's your name?"},
        ],
    )
    print("=== OpenAI SDK (non-streaming) ===")
    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
    print()


def openai_streaming():
    """OpenAI SDK — streaming."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{API_BASE}/v1", api_key=API_KEY)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count to 5."}],
        stream=True,
    )
    print("=== OpenAI SDK (streaming) ===")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def anthropic_non_streaming():
    """Anthropic SDK — non-streaming."""
    from anthropic import Anthropic

    client = Anthropic(base_url=API_BASE, api_key=API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a helpful assistant named lingg.",
        messages=[{"role": "user", "content": "Hello! What's your name?"}],
    )
    print("=== Anthropic SDK (non-streaming) ===")
    print(f"Model: {response.model}")
    for block in response.content:
        if block.type == "text":
            print(f"Response: {block.text}")
    print()


def anthropic_streaming():
    """Anthropic SDK — streaming."""
    from anthropic import Anthropic

    client = Anthropic(base_url=API_BASE, api_key=API_KEY)
    print("=== Anthropic SDK (streaming) ===")
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Count to 5."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


def print_claude_code_config():
    """Print Claude Code configuration."""
    import json

    print("=== Claude Code Configuration ===")
    print("Add to your shell environment:")
    print(f'  $env:ANTHROPIC_BASE_URL="{API_BASE}"')
    print(f'  $env:ANTHROPIC_API_KEY="{API_KEY}"')
    print(f'  $env:ANTHROPIC_MODEL="{MODEL}"')
    print()
    print("Or in ~/.claude/settings.json:")
    print(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": API_BASE,
            "ANTHROPIC_API_KEY": API_KEY,
            "ANTHROPIC_MODEL": MODEL,
        }
    }, indent=2))
    print()


if __name__ == "__main__":
    import asyncio

    print(f"API Base: {API_BASE}")
    print(f"Model: {MODEL}")
    print()

    for fn in [openai_non_streaming, openai_streaming,
               anthropic_non_streaming, anthropic_streaming]:
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} error: {e}\n")

    print_claude_code_config()
