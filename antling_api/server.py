"""
ant-ling Chat API Server
========================

Reverse-engineered from network-intercept-har.txt.

Three layers:
  1. Native ant-ling routes  — faithful replicas of chat.ant-ling.com
  2. OpenAI adapter          — /v1/chat/completions, /v1/models
  3. Anthropic adapter       — /v1/messages, /v1/models

Two modes:
  Proxy mode    — forwards to real chat.ant-ling.com (set ANTLING_BASE_URL)
  Standalone    — returns mock responses (default, no upstream needed)

Auth (for proxy mode):
  The real API uses Cookie-based auth (TLingSESSIONID JWT) + tenant-id header.
  Set ANTLING_SESSION_COOKIE in .env with the full cookie string from your browser.

Usage:
  uvicorn antling_api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from antling_api.config import (
    API_KEY,
    BASE_URL,
    HOST,
    MODEL_CACHE_TTL,
    MODELS,
    PORT,
    TENANT_ID,
    is_proxy_mode,
)
from antling_api.models import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicModel,
    AnthropicModelsResponse,
    AnthropicUsage,
    ChatRequest,
    ConfigModel,
    ConversationResponse,
    CreateConversationRequest,
    MessageItem,
    MessagesResponse,
    ModelCustomParamResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModel,
    OpenAIModelsResponse,
    OpenAIStreamResponse,
    OpenAIUsage,
    TitleResponse,
    UpdateModelCustomParamRequest,
    UpdateResponse,
)
from antling_api.upstream import collect_stream, forward_request, forward_stream

# ── App ────────────────────────────────────────

app = FastAPI(
    title="ant-ling Chat API",
    description="Reverse-engineered API for chat.ant-ling.com — OpenAI & Anthropic compatible",
    version="1.0.0",
)

# ── In-memory store (standalone mode) ──────────

_conversations: dict[str, dict[str, Any]] = {}
_messages: dict[str, list[MessageItem]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def _gen_conversation_id() -> str:
    """Generate conversation IDs matching the real format: YYYYMMDD{hash}."""
    date_prefix = datetime.now().strftime("%Y%m%d")
    return f"{date_prefix}{uuid.uuid4().hex[:16].upper()}"


# ── Model registry ───────────────────────────────
# Proxy mode:  always fetch from upstream on every request (no cache).
# Standalone:  cache with TTL, refresh automatically when expired.


class ModelRegistry:
    """Manages the model list for all endpoints.

    Proxy mode:  fetches from upstream on every call (auth headers forwarded).
    Standalone:  fetches once, caches for MODEL_CACHE_TTL seconds, then refreshes.
    Fallback:    uses hardcoded MODELS list if upstream is unreachable.
    """

    def __init__(self) -> None:
        self._cache: list[dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def get_models(self) -> list[dict[str, Any]]:
        """Return the current model list, using the appropriate strategy."""
        if is_proxy_mode():
            return await self._fetch_from_proxy()
        return await self._fetch_with_ttl()

    async def _fetch_from_proxy(self) -> list[dict[str, Any]]:
        """Proxy mode — always forward to upstream with user auth context."""
        try:
            resp = await forward_request("GET", "/meta/model/list")
            if resp.status_code < 400:
                data = resp.json()
                models = data.get("data", data) if isinstance(data, dict) else data
                if models and isinstance(models, list):
                    return models
        except Exception:
            pass
        return self._fallback()

    async def _fetch_with_ttl(self) -> list[dict[str, Any]]:
        """Standalone mode — use cache if fresh, otherwise fetch."""
        now = datetime.now(timezone.utc).timestamp()
        if self._cache is not None and (now - self._cache_time) < MODEL_CACHE_TTL:
            return self._cache

        models = await self._fetch_from_upstream()
        if models:
            self._cache = models
            self._cache_time = now
            return models

        if self._cache is not None:
            return self._cache
        return self._fallback()

    @staticmethod
    async def _fetch_from_upstream() -> list[dict[str, Any]]:
        """Fetch from the public endpoint (no auth required).

        Enriches each model with a normalized 'tools' and 'skills' list
        extracted from the upstream customParam structure.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://chat.ant-ling.com/meta/model/list")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", data) if isinstance(data, dict) else data
                    if models and isinstance(models, list):
                        return [_normalize_model(m) for m in models]
        except Exception:
            pass
        return []

    @staticmethod
    def _fallback() -> list[dict[str, Any]]:
        """Hardcoded fallback when upstream is unreachable.

        Tools and skills are derived from the upstream customParam structure
        where Switch-type params with tags "tool" or "skill" represent
        capabilities that can be toggled per conversation.
        """
        return [
            {
                "id": "Ring-2.6-1T",
                "code": "Ring-2.6-1T",
                "name": "Ring-2.6-1T",
                "displayName": "Ring 2.6 1T",
                "description": "万亿级卓越推理能力，深度理性思考的旗舰模型",
                "version": "2.6",
                "contextWindow": 131072,
                "maxTokens": 65536,
                "modal": "TEXT",
                "tags": ["深度思考"],
                "tools": ["webSearch", "currentTime", "weather", "htmlPageGenerate"],
                "skills": ["multimodalReply"],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.8, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 65536, "config": {"min": 1, "max": 65536, "step": 1}},
                    {"name": "_webSearch", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_get_current_time", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_query_weather", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_html_page_generate", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                    {"name": "_multimodal_reply", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                    {"name": "reasoningEffort", "type": "string", "defaultValue": "high", "config": {"options": [{"value": "high", "label": "快速"}, {"value": "xhigh", "label": "中等"}, {"value": "heavy", "label": "DeepThink", "tag": "Heavy"}]}},
                ],
                "supportedFeatures": ["chat", "streaming", "webSearch", "functionCall", "deepReasoning"],
                "object": "model",
            },
            {
                "id": "Ling-2.6-1T",
                "code": "Ling-2.6-1T",
                "name": "Ling-2.6-1T",
                "displayName": "Ling 2.6 1T",
                "description": "万亿级快思考旗舰，即时执行零冗余",
                "version": "2.6",
                "contextWindow": 131072,
                "maxTokens": 32768,
                "modal": "TEXT",
                "tags": ["工具调用"],
                "tools": ["webSearch", "currentTime", "weather", "htmlPageGenerate"],
                "skills": ["multimodalReply"],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.8, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 32768, "config": {"min": 1, "max": 32768, "step": 1}},
                    {"name": "_webSearch", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_get_current_time", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_query_weather", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_html_page_generate", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                    {"name": "_multimodal_reply", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                ],
                "supportedFeatures": ["chat", "streaming", "webSearch", "functionCall", "toolCall"],
                "object": "model",
            },
            {
                "id": "Ling-2.6-flash",
                "code": "Ling-2.6-flash",
                "name": "Ling-2.6-flash",
                "displayName": "Ling 2.6 Flash",
                "description": "极速激活，轻巧强大，为效率而生（104B 激活 7.4B）",
                "version": "2.6",
                "contextWindow": 131072,
                "maxTokens": 32768,
                "modal": "TEXT",
                "tags": ["工具调用"],
                "tools": ["currentTime", "weather", "newsSearch"],
                "skills": [],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.6, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 32768, "config": {"min": 1, "max": 32768, "step": 1}},
                    {"name": "_get_current_time", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_query_weather", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_news_search", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                ],
                "supportedFeatures": ["chat", "streaming", "functionCall", "toolCall"],
                "object": "model",
            },
            {
                "id": "Ling-2.5-1T",
                "code": "Ling-2.5-1T",
                "name": "Ling-2.5-1T",
                "displayName": "Ling 2.5 1T",
                "description": "万亿级思考力，兼顾深度与速度的通用语言基座（激活 63B）",
                "version": "2.5",
                "contextWindow": 131072,
                "maxTokens": 32768,
                "modal": "TEXT",
                "tags": ["工具调用"],
                "tools": ["webSearch", "currentTime", "weather", "htmlPageGenerate"],
                "skills": ["multimodalReply"],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.8, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 32768, "config": {"min": 1, "max": 32768, "step": 1}},
                    {"name": "_webSearch", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_get_current_time", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_query_weather", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_html_page_generate", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                    {"name": "_multimodal_reply", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                ],
                "supportedFeatures": ["chat", "streaming", "webSearch", "functionCall", "toolCall"],
                "object": "model",
            },
            {
                "id": "Ring-2.5-1T",
                "code": "Ring-2.5-1T",
                "name": "Ring-2.5-1T",
                "displayName": "Ring 2.5 1T",
                "description": "万亿级卓越推理能力，深度理性思考的旗舰模型",
                "version": "2.5",
                "contextWindow": 131072,
                "maxTokens": 65536,
                "modal": "TEXT",
                "tags": ["深度思考"],
                "tools": ["webSearch", "currentTime", "weather", "htmlPageGenerate"],
                "skills": ["multimodalReply"],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.8, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 65536, "config": {"min": 1, "max": 65536, "step": 1}},
                    {"name": "_webSearch", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_get_current_time", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_query_weather", "type": "boolean", "defaultValue": False, "tags": ["tool"]},
                    {"name": "_html_page_generate", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                    {"name": "_multimodal_reply", "type": "boolean", "defaultValue": False, "tags": ["skill"]},
                ],
                "supportedFeatures": ["chat", "streaming", "webSearch", "functionCall", "deepReasoning"],
                "object": "model",
            },
            {
                "id": "Ming-Omni-flash",
                "code": "Ming-Omni-flash",
                "name": "Ming-Omni-flash",
                "displayName": "Ming Omni Flash",
                "description": "支持图片、音视频输入，语音和图像输出（103B 激活 9B）",
                "version": "1.0",
                "contextWindow": 131072,
                "maxTokens": 65536,
                "modal": "MULTI_MODAL",
                "tags": ["全模态"],
                "tools": [],
                "skills": ["multimodalReply"],
                "inputFileConfig": [
                    {"mediaCategory": "IMAGE", "fileConfig": {"maxCount": 1, "size": 10485760, "suffix": ["jpeg", "png", "jpg"]}},
                    {"mediaCategory": "VIDEO", "fileConfig": {"maxCount": 1, "size": 104857600, "suffix": ["mp4", "avi", "mkv"]}},
                    {"mediaCategory": "AUDIO", "fileConfig": {"maxCount": 1, "size": 104857600, "suffix": ["wav", "mp3"], "duration": 30}},
                ],
                "multimodalCapabilities": {
                    "imageUnderstand": True,
                    "videoUnderstand": True,
                    "audioUnderstand": True,
                    "imageSplit": True,
                    "imageGenerate": False,
                    "imageEdit": False,
                },
                "supportedFeatures": ["chat", "streaming", "functionCall", "multimodal", "imageInput", "audioInput", "videoInput", "voiceOutput", "imageOutput"],
                "object": "model",
            },
            {
                "id": "AntAngelMed",
                "code": "AntAngelMed",
                "name": "AntAngelMed",
                "displayName": "AntAngelMed",
                "description": "蚂蚁·安诊儿医疗大模型",
                "version": "1.0",
                "contextWindow": 131072,
                "maxTokens": 65536,
                "modal": "TEXT",
                "tags": ["医疗大模型", "深度思考"],
                "tools": [],
                "skills": [],
                "customParam": [
                    {"name": "systemPrompt", "type": "string", "defaultValue": ""},
                    {"name": "temperature", "type": "double", "defaultValue": 0.8, "config": {"min": 0, "max": 2, "step": 0.1}},
                    {"name": "topK", "type": "integer", "defaultValue": 20, "config": {"min": 1, "max": 200, "step": 1}},
                    {"name": "topP", "type": "double", "defaultValue": 0.95, "config": {"min": 0.1, "max": 1, "step": 0.1}},
                    {"name": "maxCompletionTokens", "type": "integer", "defaultValue": 65536, "config": {"min": 1, "max": 65536, "step": 1}},
                ],
                "supportedFeatures": ["chat", "streaming", "functionCall", "deepReasoning", "medical"],
                "object": "model",
            },
        ]


def _normalize_model(model: dict[str, Any]) -> dict[str, Any]:
    """Normalize an upstream model object for our API.

    Extracts tools/skills from customParam and adds them as top-level
    `tools` and `skills` arrays for easy client consumption.
    Strips the leading underscore from tool/skill param names.
    """
    custom_params = model.get("customParam", [])
    tools: list[str] = []
    skills: list[str] = []

    for param in custom_params:
        param_tags = param.get("tags", [])
        name: str = param.get("name", "")
        if not name:
            continue
        if "tool" in param_tags:
            tools.append(name.lstrip("_"))
        elif "skill" in param_tags:
            skills.append(name.lstrip("_"))

    model["tools"] = tools
    model["skills"] = skills
    return model


# Singleton instance used across all endpoints
_model_registry = ModelRegistry()


@app.on_event("startup")
async def _startup_fetch_models():
    """Pre-fetch the model list on startup so the first request is fast."""
    await _model_registry.get_models()


# ═══════════════════════════════════════════════
# 1. Native ant-ling routes
# ═══════════════════════════════════════════════


@app.get("/meta/model/list")
async def list_models():
    """GET /meta/model/list — list available models.

    Proxy mode:  forwards to upstream with user auth context (always fresh).
    Standalone:  returns TTL-cached list (refreshes automatically).

    Each model includes `tools` and `skills` arrays extracted from the
    upstream customParam structure (Switch-type params tagged "tool" or "skill").
    """
    models = await _model_registry.get_models()
    return {"object": "list", "data": models}


@app.get("/meta/model/capabilities")
async def list_model_capabilities():
    """GET /meta/model/capabilities — aggregated capabilities catalog.

    Returns a deduplicated list of all tools, skills, and parameters
    available across all models, plus per-model capability mapping.
    Useful for clients that need to know what features are available
    before selecting a model.
    """
    models = await _model_registry.get_models()

    all_tools: set[str] = set()
    all_skills: set[str] = set()
    all_params: set[str] = set()
    per_model: dict[str, dict[str, list[str]]] = {}

    for m in models:
        code = m.get("code", m["id"])
        tools = m.get("tools", [])
        skills = m.get("skills", [])
        params = [p["name"] for p in m.get("customParam", []) if "tags" not in p or not p["tags"]]

        all_tools.update(tools)
        all_skills.update(skills)
        all_params.update(params)
        per_model[code] = {"tools": tools, "skills": skills, "params": params}

    return {
        "tools": sorted(all_tools),
        "skills": sorted(all_skills),
        "parameters": sorted(all_params),
        "models": per_model,
    }


@app.post("/meta/conversation/create", response_model=ConversationResponse)
async def create_conversation(req: CreateConversationRequest):
    """POST /meta/conversation/create — create a new conversation."""
    if is_proxy_mode():
        try:
            resp = await forward_request(
                "POST", "/meta/conversation/create", body=req.model_dump()
            )
            if resp.status_code < 400:
                return ConversationResponse(**resp.json())
        except Exception:
            pass

    conv_id = _gen_conversation_id()
    _conversations[conv_id] = {
        "conversationId": conv_id,
        "title": req.query[:30],
        "createdAt": _now(),
        "model": "Ring-2.6-1T",
    }
    _messages[conv_id] = []
    return ConversationResponse(conversationId=conv_id, title=req.query[:30], createdAt=_now())


@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    """
    POST /api/v1/chat — main chat endpoint.
    Returns SSE (Server-Sent Events) stream.
    """
    body = req.model_dump()
    if not req.conversationId:
        body["conversationId"] = _gen_conversation_id()

    if not is_proxy_mode():
        # Store user message
        conv_id = body["conversationId"]
        if conv_id not in _messages:
            _messages[conv_id] = []
        _messages[conv_id].append(
            MessageItem(role="user", content=req.input, id=_gen_id("msg_"))
        )

    if is_proxy_mode():
        async def upstream_stream() -> AsyncGenerator[str, None]:
            async for line in forward_stream("POST", "/api/v1/chat", body=body):
                yield f"{line}\n"

        return StreamingResponse(
            upstream_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "x-webgw-protocol": "chunked",
            },
        )

    # Standalone mock stream
    async def mock_stream() -> AsyncGenerator[str, None]:
        response_text = (
            f"Hello! I'm lingg, an AI assistant. "
            f"You said: '{req.input}'. How can I help you today?"
        )
        words = response_text.split()
        msg_id = _gen_id("msg_")

        for i, word in enumerate(words):
            chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

        conv_id = body.get("conversationId", "")
        if conv_id in _messages:
            _messages[conv_id].append(
                MessageItem(role="assistant", content=response_text, id=msg_id)
            )

    return StreamingResponse(
        mock_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"},
    )


@app.get("/meta/message/messages", response_model=MessagesResponse)
async def get_messages(
    conversationId: str = Query(...),
    currentPage: int = Query(1),
    pageSize: int = Query(20),
):
    """GET /meta/message/messages — get messages for a conversation (paginated)."""
    if is_proxy_mode():
        try:
            resp = await forward_request(
                "GET",
                "/meta/message/messages",
                params={
                    "conversationId": conversationId,
                    "currentPage": currentPage,
                    "pageSize": pageSize,
                },
            )
            if resp.status_code < 400:
                return MessagesResponse(**resp.json())
        except Exception:
            pass

    msgs = _messages.get(conversationId, [])
    total = len(msgs)
    start = (currentPage - 1) * pageSize
    return MessagesResponse(
        conversationId=conversationId,
        messages=msgs[start:start + pageSize],
        currentPage=currentPage,
        pageSize=pageSize,
        total=total,
    )


@app.get("/meta/conversation/modelCustomParam", response_model=ModelCustomParamResponse)
async def get_model_custom_param(conversationId: str = Query(...)):
    """GET /meta/conversation/modelCustomParam — get model config."""
    if is_proxy_mode():
        try:
            resp = await forward_request(
                "GET",
                "/meta/conversation/modelCustomParam",
                params={"conversationId": conversationId},
            )
            if resp.status_code < 400:
                return ModelCustomParamResponse(**resp.json())
        except Exception:
            pass

    return ModelCustomParamResponse(
        conversationId=conversationId,
        modelCustomParam=ConfigModel(),
    )


@app.post("/meta/conversation/updateModelCustomParam", response_model=UpdateResponse)
async def update_model_custom_param(req: UpdateModelCustomParamRequest):
    """POST /meta/conversation/updateModelCustomParam — update model config."""
    if is_proxy_mode():
        try:
            resp = await forward_request(
                "POST",
                "/meta/conversation/updateModelCustomParam",
                body=req.model_dump(),
            )
            if resp.status_code < 400:
                return UpdateResponse(**resp.json())
        except Exception:
            pass

    return UpdateResponse(success=True, message="Model parameters updated")


@app.post("/meta/conversation/{conversation_id}/title", response_model=TitleResponse)
async def generate_title(conversation_id: str):
    """POST /meta/conversation/{id}/title — auto-generate conversation title."""
    if is_proxy_mode():
        try:
            resp = await forward_request(
                "POST", f"/meta/conversation/{conversation_id}/title"
            )
            if resp.status_code < 400:
                return TitleResponse(**resp.json())
        except Exception:
            pass

    msgs = _messages.get(conversation_id, [])
    title = msgs[0].content[:30] if msgs else "New Chat"
    return TitleResponse(conversation_id=conversation_id, title=title, success=True)


# ═══════════════════════════════════════════════
# 2. OpenAI-compatible adapter
# ═══════════════════════════════════════════════

OPENAI_MODEL_MAP = {
    # Ring series (deep reasoning)
    "Ring-2.6-1T": "Ring-2.6-1T",
    "Ring-2.6": "Ring-2.6-1T",
    "Ring-2.5-1T": "Ring-2.5-1T",
    # Ling series (fast / tool-calling)
    "Ling-2.6-1T": "Ling-2.6-1T",
    "Ling-2.6-flash": "Ling-2.6-flash",
    "Ling-2.5-1T": "Ling-2.5-1T",
    # Omni (multimodal)
    "Ming-Omni-flash": "Ming-Omni-flash",
    # Med
    "AntAngelMed": "AntAngelMed",
    # OpenAI aliases → Ring-2.6-1T (default)
    "gpt-4": "Ring-2.6-1T",
    "gpt-4o": "Ring-2.6-1T",
    "gpt-3.5-turbo": "Ring-2.6-1T",
}


@app.get("/v1/models", response_model=OpenAIModelsResponse)
async def openai_list_models():
    """OpenAI-compatible model listing.

    Uses the human-readable `code` field (e.g. "Ring-2.6-1T") rather than
    the internal upstream ID (e.g. "20260310ANIN00015080").
    """
    models = await _model_registry.get_models()
    return OpenAIModelsResponse(
        data=[OpenAIModel(id=m.get("code", m["id"]), owned_by="antling") for m in models]
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    """
    OpenAI-compatible /v1/chat/completions.
    Translates to native /api/v1/chat.
    Supports streaming and non-streaming.
    """
    model = OPENAI_MODEL_MAP.get(req.model, "Ring-2.6-1T")

    system_prompt = ""
    user_parts: list[str] = []
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content if isinstance(msg.content, str) else ""
        elif msg.role == "user":
            if isinstance(msg.content, str):
                user_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_parts.append(part.get("text", ""))

    user_input = "\n".join(user_parts) if user_parts else "Hello"

    config = ConfigModel(
        systemPrompt=system_prompt,
        temperature=req.temperature if req.temperature is not None else 0.8,
        topP=req.top_p if req.top_p is not None else 0.95,
        maxCompletionTokens=req.max_tokens if req.max_tokens else 65536,
    )
    chat_req = ChatRequest(input=user_input, model=model, config=config)

    if req.stream:
        return await _openai_stream(chat_req, model)
    return await _openai_non_stream(chat_req, model)


async def _openai_non_stream(chat_req: ChatRequest, model: str) -> OpenAIChatResponse:
    if is_proxy_mode():
        try:
            full = await collect_stream("/api/v1/chat", chat_req.model_dump())
            return OpenAIChatResponse(
                model=model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=full),
                        finish_reason="stop",
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=len(chat_req.input.split()),
                    completion_tokens=len(full.split()),
                    total_tokens=len(chat_req.input.split()) + len(full.split()),
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    # Standalone
    text = f"Hello! I'm lingg. You said: '{chat_req.input}'. How can I help?"
    return OpenAIChatResponse(
        model=model,
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=len(chat_req.input.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(chat_req.input.split()) + len(text.split()),
        ),
    )


async def _openai_stream(chat_req: ChatRequest, model: str) -> StreamingResponse:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now(timezone.utc).timestamp())

    if is_proxy_mode():
        async def upstream() -> AsyncGenerator[str, None]:
            try:
                async for line in forward_stream("POST", "/api/v1/chat", chat_req.model_dump()):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data_str)
                            chunk["id"] = response_id
                            chunk["object"] = "chat.completion.chunk"
                            chunk["created"] = created
                            chunk["model"] = model
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            upstream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"},
        )

    # Standalone mock — send all chunks immediately (no sleeps)
    async def mock() -> AsyncGenerator[str, None]:
        text = f"Hello! I'm lingg. You said: '{chat_req.input}'. How can I help?"
        for i, word in enumerate(text.split()):
            delta = OpenAIMessage(role="assistant",
                                 content=word + (" " if i < len(text.split()) - 1 else ""))
            choice = OpenAIChoice(index=0, delta=delta)
            chunk = OpenAIStreamResponse(id=response_id, created=created, model=model, choices=[choice])
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        final_delta = OpenAIMessage(role="assistant")
        final_choice = OpenAIChoice(index=0, delta=final_delta, finish_reason="stop")
        final = OpenAIStreamResponse(id=response_id, created=created, model=model, choices=[final_choice])
        yield f"data: {final.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        mock(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"},
    )


# ═══════════════════════════════════════════════
# 3. Anthropic-compatible adapter
# ═══════════════════════════════════════════════

ANTHROPIC_MODEL_MAP = {
    # Ring series (deep reasoning)
    "Ring-2.6-1T": "Ring-2.6-1T",
    "Ring-2.6": "Ring-2.6-1T",
    "Ring-2.5-1T": "Ring-2.5-1T",
    # Ling series (fast / tool-calling)
    "Ling-2.6-1T": "Ling-2.6-1T",
    "Ling-2.6-flash": "Ling-2.6-flash",
    "Ling-2.5-1T": "Ling-2.5-1T",
    # Omni (multimodal)
    "Ming-Omni-flash": "Ming-Omni-flash",
    # Med
    "AntAngelMed": "AntAngelMed",
    # Anthropic aliases → Ring-2.6-1T (default)
    "claude-3-5-sonnet": "Ring-2.6-1T",
    "claude-3-5-sonnet-20241022": "Ring-2.6-1T",
    "claude-3-7-sonnet": "Ring-2.6-1T",
    "claude-3-7-sonnet-20250219": "Ring-2.6-1T",
    "claude-sonnet-4-6": "Ring-2.6-1T",
}


@app.get("/v1/models", response_model=AnthropicModelsResponse)
async def anthropic_list_models():
    """Anthropic-compatible model listing.

    Uses the human-readable `code` field for the model ID.
    """
    models = await _model_registry.get_models()
    return AnthropicModelsResponse(
        data=[
            AnthropicModel(
                id=m.get("code", m["id"]),
                display_name=m.get("displayName", m.get("name", m["id"])),
            )
            for m in models
        ]
    )


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    """
    Anthropic-compatible /v1/messages.
    Translates to native /api/v1/chat.
    Supports streaming and non-streaming.
    """
    model = ANTHROPIC_MODEL_MAP.get(req.model, "Ring-2.6-1T")

    system_prompt = req.system or ""
    user_parts: list[str] = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            user_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    user_parts.append(block.get("text", ""))

    user_input = "\n".join(user_parts) if user_parts else "Hello"

    config = ConfigModel(
        systemPrompt=system_prompt,
        temperature=req.temperature if req.temperature is not None else 0.8,
        topP=req.top_p if req.top_p is not None else 0.95,
        topK=req.top_k if req.top_k is not None else 20,
        maxCompletionTokens=req.max_tokens if req.max_tokens else 65536,
    )
    chat_req = ChatRequest(input=user_input, model=model, config=config)

    if req.stream:
        return await _anthropic_stream(chat_req, model)
    return await _anthropic_non_stream(chat_req, model)


async def _anthropic_non_stream(
    chat_req: ChatRequest, model: str
) -> AnthropicMessagesResponse:
    if is_proxy_mode():
        try:
            full = await collect_stream("/api/v1/chat", chat_req.model_dump())
            return AnthropicMessagesResponse(
                model=model,
                content=[AnthropicContentBlock(type="text", text=full)],
                stop_reason="end_turn",
                usage=AnthropicUsage(
                    input_tokens=len(chat_req.input.split()),
                    output_tokens=len(full.split()),
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    text = f"Hello! I'm lingg. You said: '{chat_req.input}'. How can I help?"
    return AnthropicMessagesResponse(
        model=model,
        content=[AnthropicContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=AnthropicUsage(
            input_tokens=len(chat_req.input.split()),
            output_tokens=len(text.split()),
        ),
    )


async def _anthropic_stream(chat_req: ChatRequest, model: str) -> StreamingResponse:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if is_proxy_mode():
        async def upstream() -> AsyncGenerator[str, None]:
            msg_start = json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
            yield f"event: message_start\ndata: {msg_start}\n\n"
            full = ""
            try:
                idx = 0
                async for line in forward_stream("POST", "/api/v1/chat", chat_req.model_dump()):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            for c in chunk.get("choices", []):
                                content = c.get("delta", {}).get("content", "")
                                if content:
                                    full += content
                                    delta_payload = json.dumps({
                                        "type": "content_block_delta", "index": idx,
                                        "delta": {"type": "text_delta", "text": content},
                                    })
                                    yield f"event: content_block_delta\ndata: {delta_payload}\n\n"
                                    idx += 1
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})
                yield f"event: error\ndata: {err}\n\n"
                return

            yield (
                f"event: content_block_stop\ndata: "
                f"{json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            )
            md = json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": len(full.split())},
            })
            yield f"event: message_delta\ndata: {md}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(
            upstream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "anthropic-version": "2023-06-01",
            },
        )

    # Standalone mock
    async def mock() -> AsyncGenerator[str, None]:
        text = f"Hello! I'm lingg. You said: '{chat_req.input}'. How can I help?"
        words = text.split()

        msg_start = json.dumps({
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield f"event: message_start\ndata: {msg_start}\n\n"

        cb_start = json.dumps({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield f"event: content_block_start\ndata: {cb_start}\n\n"

        for word in words:
            cb_delta = json.dumps({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": word + " "},
            })
            yield f"event: content_block_delta\ndata: {cb_delta}\n\n"

        yield (
            f"event: content_block_stop\ndata: "
            f"{json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        )
        md = json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(words)},
        })
        yield f"event: message_delta\ndata: {md}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    return StreamingResponse(
        mock(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "anthropic-version": "2023-06-01",
        },
    )


# ═══════════════════════════════════════════════
# 4. Health & info
# ═══════════════════════════════════════════════


@app.get("/")
async def root():
    mode = f"proxy → {BASE_URL}" if is_proxy_mode() else "standalone (mock)"
    models = await _model_registry.get_models()
    return {
        "service": "ant-ling Chat API",
        "version": "1.0.0",
        "mode": mode,
        "models": [m.get("code", m["id"]) for m in models],
        "endpoints": {
            "native": [
                "GET  /meta/model/list",
                "GET  /meta/model/capabilities",
                "POST /meta/conversation/create",
                "POST /api/v1/chat",
                "GET  /meta/message/messages",
                "GET  /meta/conversation/modelCustomParam",
                "POST /meta/conversation/updateModelCustomParam",
                "POST /meta/conversation/{id}/title",
            ],
            "openai": [
                "GET  /v1/models",
                "POST /v1/chat/completions",
            ],
            "anthropic": [
                "GET  /v1/models",
                "POST /v1/messages",
            ],
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now()}


# ═══════════════════════════════════════════════
# 5. Run
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    mode = f"proxy → {BASE_URL}" if is_proxy_mode() else "standalone (mock)"
    print(f"""
╔══════════════════════════════════════════════════════╗
║           ant-ling Chat API Server                   ║
╠══════════════════════════════════════════════════════╣
║  URL:     http://{HOST}:{PORT}                        ║
║  Mode:    {mode:<43s} ║
║  Models:  {', '.join(MODELS):<43s} ║
╠══════════════════════════════════════════════════════╣
║  OpenAI:     http://{HOST}:{PORT}/v1/...              ║
║  Anthropic:  http://{HOST}:{PORT}/v1/messages         ║
║  Native:     http://{HOST}:{PORT}/api/v1/chat         ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("antling_api.server:app", host=HOST, port=PORT)
