"""
Upstream proxy module.

Handles forwarding requests to the real chat.ant-ling.com server
with proper authentication (cookies + tenant-id header).
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import httpx

from antling_api.config import BASE_URL, API_KEY, SESSION_COOKIE, TENANT_ID


def _build_headers(stream: bool = False) -> dict[str, str]:
    """Build request headers matching the real API."""
    hdrs: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "tenant-id": TENANT_ID,
        "x-webgw-protocol": "chunked",
    }
    if SESSION_COOKIE:
        hdrs["Cookie"] = SESSION_COOKIE
    if API_KEY:
        hdrs["Authorization"] = f"Bearer {API_KEY}"
    return hdrs


async def forward_request(
    method: str,
    path: str,
    body: dict | None = None,
    params: dict | None = None,
) -> httpx.Response:
    """Forward a single request to the upstream server."""
    url = f"{BASE_URL}{path}"
    hdrs = _build_headers(stream=False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        if method.upper() == "GET":
            return await client.get(url, params=params, headers=hdrs)
        elif method.upper() == "POST":
            return await client.post(url, json=body, params=params, headers=hdrs)
        elif method.upper() == "PUT":
            return await client.put(url, json=body, params=params, headers=hdrs)
        elif method.upper() == "DELETE":
            return await client.delete(url, params=params, headers=hdrs)
        else:
            raise ValueError(f"Unsupported method: {method}")


async def forward_stream(
    method: str,
    path: str,
    body: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Forward a request and yield SSE lines from the upstream response."""
    url = f"{BASE_URL}{path}"
    hdrs = _build_headers(stream=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(method.upper(), url, json=body, headers=hdrs) as resp:
            async for line in resp.aiter_lines():
                yield line


async def collect_stream(path: str, body: dict | None = None) -> str:
    """Forward a streaming request and collect the full content."""
    full_content = ""
    async for line in forward_stream("POST", path, body):
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
            except json.JSONDecodeError:
                pass
    return full_content
