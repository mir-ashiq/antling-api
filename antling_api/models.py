"""
Pydantic models for the ant-ling Chat API.

Covers three layers:
  1. Native ant-ling endpoints (reverse-engineered from HAR capture)
  2. OpenAI-compatible adapter (/v1/chat/completions)
  3. Anthropic-compatible adapter (/v1/messages)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════
# 1. Native ant-ling models
# ═══════════════════════════════════════════════


class ConfigModel(BaseModel):
    """Model configuration sent with chat requests."""
    systemPrompt: str = ""
    temperature: float = 0.8
    topK: int = 20
    topP: float = 0.95
    maxCompletionTokens: int = 65536
    _webSearch: bool = False
    _get_current_time: bool = False
    _query_weather: bool = False
    _html_page_generate: bool = False
    _multimodal_reply: bool = False
    reasoningEffort: Literal["low", "medium", "high"] = "high"


class ChatFile(BaseModel):
    """File attachment (schema observed in request)."""
    url: str = ""
    name: str = ""
    mimeType: str = ""


class ChatRequest(BaseModel):
    """POST /api/v1/chat -- main chat endpoint (returns SSE stream)."""
    input: str
    conversationId: str = ""
    model: str = "Ring-2.6-1T"
    config: ConfigModel = Field(default_factory=ConfigModel)
    files: list[ChatFile] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    """POST /meta/conversation/create"""
    query: str


class ConversationResponse(BaseModel):
    """Response from create conversation."""
    conversationId: str
    title: str = ""
    createdAt: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model: str = "Ring-2.6-1T"


class MessageItem(BaseModel):
    """A single message in a conversation."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Literal["user", "assistant", "system"] = "user"
    content: str = ""
    createdAt: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MessagesResponse(BaseModel):
    """GET /meta/message/messages"""
    conversationId: str = ""
    messages: list[MessageItem] = Field(default_factory=list)
    currentPage: int = 1
    pageSize: int = 20
    total: int = 0


class ModelCustomParamResponse(BaseModel):
    """GET /meta/conversation/modelCustomParam"""
    conversationId: str = ""
    modelCustomParam: ConfigModel = Field(default_factory=ConfigModel)


class UpdateModelCustomParamRequest(BaseModel):
    """POST /meta/conversation/updateModelCustomParam"""
    conversationId: str
    modelCustomParam: ConfigModel


class UpdateResponse(BaseModel):
    """Generic update response."""
    success: bool = True
    message: str = "updated"


class TitleResponse(BaseModel):
    """POST /meta/conversation/{id}/title"""
    conversationId: str = ""
    title: str = ""
    success: bool = True


class ModelListItem(BaseModel):
    """A single model entry in /meta/model/list response."""
    id: str
    name: str = ""
    displayName: str = ""
    description: str = ""
    version: str = ""
    contextWindow: int = 131072
    maxTokens: int = 65536
    supportedFeatures: list[str] = Field(default_factory=list)
    pricing: dict[str, Any] = Field(default_factory=dict)
    object: str = "model"


class ModelListResponse(BaseModel):
    """GET /meta/model/list"""
    data: list[ModelListItem] = Field(default_factory=list)
    object: str = "list"


# ═══════════════════════════════════════════════
# 2. OpenAI-compatible models
# ═══════════════════════════════════════════════


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChatRequest(BaseModel):
    model: str = "Ring-2.6-1T"
    messages: list[OpenAIMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    user: str | None = None


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage | None = None
    delta: OpenAIMessage | None = None
    finish_reason: str | None = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    model: str = "Ring-2.6-1T"
    choices: list[OpenAIChoice] = Field(default_factory=list)
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)


class OpenAIStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion.chunk"
    created: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    model: str = "Ring-2.6-1T"
    choices: list[OpenAIChoice] = Field(default_factory=list)


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    owned_by: str = "antling"


class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIModel] = Field(default_factory=list)


# ═══════════════════════════════════════════════
# 3. Anthropic-compatible models
# ═══════════════════════════════════════════════


class AnthropicContentBlock(BaseModel):
    type: Literal["text", "tool_use", "tool_result"] = "text"
    text: str = ""
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    tool_use_id: str | None = None


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]] = ""


class AnthropicMessagesRequest(BaseModel):
    model: str = "Ring-2.6-1T"
    messages: list[AnthropicMessage]
    system: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    model: str = "Ring-2.6-1T"
    content: list[AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicModel(BaseModel):
    id: str
    display_name: str
    type: str = "model"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AnthropicModelsResponse(BaseModel):
    data: list[AnthropicModel] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None
