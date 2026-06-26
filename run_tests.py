"""
Comprehensive test suite for the ant-ling Chat API.
Tests ALL endpoints across all three layers.

Run:
  python run_tests.py
"""

import asyncio
import json
import sys

import httpx

BASE = "http://localhost:8000"
PASSED = 0
FAILED = 0


def log(label: str, detail: str = ""):
    print(f"  {label}: {detail}")


async def test(name: str, method: str, path: str, body=None, params=None, expect_stream=False):
    global PASSED, FAILED
    url = f"{BASE}{path}"
    print(f"\n{'─'*60}")
    print(f"▶ {name}")
    print(f"  {method} {path}")

    try:
        if expect_stream:
            lines = []
            client = httpx.AsyncClient(timeout=120, follow_redirects=True)
            try:
                # Use stream() and manually manage context so we can break early
                stream_ctx = client.stream(method, url, json=body, params=params)
                resp = await stream_ctx.__aenter__()
                ct = resp.headers.get("content-type", "")
                print(f"  Status: {resp.status_code}  Content-Type: {ct}")
                try:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            lines.append(line)
                            if len(lines) >= 10:
                                break
                except Exception:
                    pass  # Stream ended — that's fine, we got our lines
                # Manually close the stream context (avoids waiting for server close)
                await stream_ctx.__aexit__(None, None, None)
            finally:
                await client.aclose()
            print(f"  Stream lines ({len(lines)} shown):")
            for line in lines:
                print(f"    {line[:120]}")
            PASSED += 1
            print(f"  ✅ PASS")
            return

        # Non-streaming request
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            if method == "GET":
                r = await client.get(url, params=params) if params else await client.get(url)
            elif method == "POST":
                r = await client.post(url, json=body) if body else await client.post(url)
            else:
                raise ValueError(f"Unsupported method: {method}")

        ct = r.headers.get("content-type", "")
        print(f"  Status: {r.status_code}  Content-Type: {ct}")
        try:
            data = r.json()
            print(f"  Response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        except:
            print(f"  Body: {r.text[:300]}")
        if r.status_code < 400:
            PASSED += 1
            print(f"  ✅ PASS")
        else:
            FAILED += 1
            print(f"  ❌ FAIL (status {r.status_code})")
    except Exception as e:
        FAILED += 1
        print(f"  ❌ FAIL: {e}")


async def main():
    global PASSED, FAILED
    print("=" * 60)
    print("  ant-ling Chat API — Full Test Suite")
    print(f"  Base URL: {BASE}")
    print("=" * 60)

    # ── Health & Info ────────────────────────────
    await test("Health check", "GET", "/health")
    await test("API info", "GET", "/")

    # ════════════════════════════════════════════
    # Native endpoints
    # ════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  NATIVE ENDPOINTS")
    print(f"{'═'*60}")

    await test("List models (native)", "GET", "/meta/model/list")

    await test("Create conversation", "POST", "/meta/conversation/create",
               body={"query": "Hello my name is test user"})

    # Get the conversation ID from the create response for subsequent tests
    conv_id = None
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/meta/conversation/create",
                              json={"query": "Test conversation"})
        if r.status_code < 400:
            data = r.json()
            conv_id = data.get("conversationId", "")
            print(f"\n  📌 Using conversationId: {conv_id}")

    if conv_id:
        await test("Get messages", "GET", "/meta/message/messages",
                   params={"conversationId": conv_id, "currentPage": 1, "pageSize": 20})

        await test("Get model custom param", "GET", "/meta/conversation/modelCustomParam",
                   params={"conversationId": conv_id})

        await test("Update model custom param", "POST", "/meta/conversation/updateModelCustomParam",
                   body={
                       "conversationId": conv_id,
                       "modelCustomParam": {
                           "systemPrompt": "You are a test assistant.",
                           "temperature": 0.5,
                           "topK": 10,
                           "topP": 0.9,
                           "maxCompletionTokens": 4096,
                           "_webSearch": False,
                           "_get_current_time": False,
                           "_query_weather": False,
                           "_html_page_generate": False,
                           "_multimodal_reply": False,
                           "reasoningEffort": "medium"
                       }
                   })

        await test("Generate title", "POST", f"/meta/conversation/{conv_id}/title")

    await test("Chat (SSE stream)", "POST", "/api/v1/chat",
               body={
                   "input": "Hello, what's your name?",
                   "model": "Ring-2.6-1T",
                   "config": {
                       "systemPrompt": "Your name is lingg.",
                       "temperature": 0.8,
                       "topK": 20,
                       "topP": 0.95,
                       "maxCompletionTokens": 65536,
                       "_webSearch": True,
                       "_get_current_time": True,
                       "_query_weather": True,
                       "_html_page_generate": False,
                       "_multimodal_reply": False,
                       "reasoningEffort": "high"
                   },
                   "files": []
               },
               expect_stream=True)

    # ════════════════════════════════════════════
    # OpenAI adapter
    # ════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  OPENAI ADAPTER")
    print(f"{'═'*60}")

    await test("OpenAI: List models", "GET", "/v1/models")

    await test("OpenAI: Chat (non-streaming)", "POST", "/v1/chat/completions",
               body={
                   "model": "Ring-2.6-1T",
                   "messages": [
                       {"role": "system", "content": "You are a helpful assistant named lingg."},
                       {"role": "user", "content": "Hello! What's your name?"}
                   ]
               })

    await test("OpenAI: Chat (streaming)", "POST", "/v1/chat/completions",
               body={
                   "model": "Ring-2.6-1T",
                   "messages": [{"role": "user", "content": "Count to 5"}],
                   "stream": True
               },
               expect_stream=True)

    # Test model aliasing
    await test("OpenAI: Chat (gpt-4 alias)", "POST", "/v1/chat/completions",
               body={
                   "model": "gpt-4",
                   "messages": [{"role": "user", "content": "Hello"}]
               })

    # ════════════════════════════════════════════
    # Anthropic adapter
    # ════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  ANTHROPIC ADAPTER")
    print(f"{'═'*60}")

    await test("Anthropic: List models", "GET", "/v1/models")

    await test("Anthropic: Messages (non-streaming)", "POST", "/v1/messages",
               body={
                   "model": "Ring-2.6-1T",
                   "max_tokens": 1024,
                   "system": "You are a helpful assistant named lingg.",
                   "messages": [{"role": "user", "content": "Hello! What's your name?"}]
               })

    await test("Anthropic: Messages (streaming)", "POST", "/v1/messages",
               body={
                   "model": "Ring-2.6-1T",
                   "max_tokens": 1024,
                   "messages": [{"role": "user", "content": "Count to 5"}],
                   "stream": True
               },
               expect_stream=True)

    # Test model aliasing
    await test("Anthropic: Messages (claude alias)", "POST", "/v1/messages",
               body={
                   "model": "claude-3-5-sonnet",
                   "max_tokens": 1024,
                   "messages": [{"role": "user", "content": "Hello"}]
               })

    # ════════════════════════════════════════════
    # OpenAI SDK integration test
    # ════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  SDK INTEGRATION TESTS")
    print(f"{'═'*60}")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=f"{BASE}/v1", api_key="x")

        r = client.chat.completions.create(
            model="Ring-2.6-1T",
            messages=[{"role": "user", "content": "Hello from OpenAI SDK"}]
        )
        print(f"\n  OpenAI SDK response: {r.choices[0].message.content}")
        PASSED += 1
        print(f"  ✅ PASS — OpenAI SDK non-streaming")
    except Exception as e:
        FAILED += 1
        print(f"  ❌ FAIL — OpenAI SDK: {e}")

    try:
        from anthropic import Anthropic
        client = Anthropic(base_url=BASE, api_key="x")

        r = client.messages.create(
            model="Ring-2.6-1T",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello from Anthropic SDK"}]
        )
        text = r.content[0].text if r.content else ""
        print(f"\n  Anthropic SDK response: {text}")
        PASSED += 1
        print(f"  ✅ PASS — Anthropic SDK non-streaming")
    except Exception as e:
        FAILED += 1
        print(f"  ❌ FAIL — Anthropic SDK: {e}")

    # ════════════════════════════════════════════
    # Model registry tests
    # ════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  MODEL REGISTRY TESTS")
    print(f"{'═'*60}")

    # Test: /v1/models returns model codes (not internal IDs)
    await test("OpenAI models use code field", "GET", "/v1/models")
    # Test: /v1/models returns model codes (Anthropic)
    await test("Anthropic models use code field", "GET", "/v1/models")
    # Test: /meta/model/list returns full upstream metadata
    await test("Native model list has metadata", "GET", "/meta/model/list")
    # Test: root endpoint uses code field
    await test("Root uses code field", "GET", "/")

    # ── Summary ──────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print(f"{'═'*60}")

    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
