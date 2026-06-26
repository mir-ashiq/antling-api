# ant-ling Chat API

Reverse-engineered Python API for [chat.ant-ling.com](https://chat.ant-ling.com) — compatible with the **OpenAI Python SDK** and **Claude Code / Anthropic SDK**.

## Features

- **Native routes** — faithful replicas of all 7 chat.ant-ling.com endpoints
- **OpenAI adapter** — `/v1/chat/completions`, `/v1/models` (drop-in replacement)
- **Anthropic adapter** — `/v1/messages`, `/v1/messages?beta=true` (Claude Code compatible)
- **Proxy mode** — forwards to the real server with cookie auth
- **Standalone mode** — mock responses for testing (no upstream needed)

## Quick Start

### Option 1: Docker (recommended)

```powershell
# Clone and start
git clone https://github.com/mir-ashiq/antling-api.git
cd antling-api
docker compose up --build

# Test
Invoke-RestMethod -Uri "http://localhost:8000/v1/models"
```

### Option 2: Local Python

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (standalone/mock mode)
uvicorn antling_api.server:app --reload --port 8000

# 3. Test it
Invoke-RestMethod -Uri "http://localhost:8000/v1/models"
```

## Authentication

The real chat.ant-ling.com API uses **two** auth mechanisms:

| Mechanism | Type | Required | How to get it |
|-----------|------|----------|---------------|
| `TLingSESSIONID` | Cookie (JWT) | ✅ Yes | Browser DevTools → Network → Cookie header |
| `tenant-id` | HTTP Header | ✅ Yes | Browser DevTools → Network → Request Headers |

**To set up proxy mode:**

1. Open [chat.ant-ling.com](https://chat.ant-ling.com) in your browser and log in
2. Open DevTools → Network tab → click any request
3. Copy the full `Cookie` header value
4. Paste it in `.env` as `ANTLING_SESSION_COOKIE`

```powershell
# .env
ANTLING_BASE_URL=https://chat.ant-ling.com
ANTLING_SESSION_COOKIE=TLingSESSIONID=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxx; jsh_t_c_e=xxxx; spanner=xxxx
ANTLING_TENANT_ID=20260523LTJY01501627
```

## Usage

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required",
)

# Non-streaming
response = client.chat.completions.create(
    model="Ring-2.6-1T",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Ring-2.6-1T",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Claude Code

Add to your environment:

```powershell
$env:ANTHROPIC_BASE_URL="http://localhost:8000"
$env:ANTHROPIC_API_KEY="sk-no-key-required"
$env:ANTHROPIC_MODEL="Ring-2.6-1T"
```

Or in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_API_KEY": "sk-no-key-required",
    "ANTHROPIC_MODEL": "Ring-2.6-1T"
  }
}
```

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="sk-no-key-required",
)

response = client.messages.create(
    model="Ring-2.6-1T",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### Native API (direct HTTP)

```powershell
# Create conversation
Invoke-RestMethod -Uri "http://localhost:8000/meta/conversation/create" `
  -Method POST -ContentType "application/json" `
  -Body '{"query":"Hello world"}'

# List models
Invoke-RestMethod -Uri "http://localhost:8000/meta/model/list"

# Get messages
Invoke-RestMethod -Uri "http://localhost:8000/meta/message/messages?conversationId=xxx&currentPage=1&pageSize=20"
```

## API Endpoints

### Native (reverse-engineered)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/meta/model/list` | List available models |
| `POST` | `/meta/conversation/create` | Create new conversation |
| `POST` | `/api/v1/chat` | Send message (SSE stream) |
| `GET` | `/meta/message/messages` | Get messages (paginated) |
| `GET` | `/meta/conversation/modelCustomParam` | Get model config |
| `POST` | `/meta/conversation/updateModelCustomParam` | Update model config |
| `POST` | `/meta/conversation/{id}/title` | Auto-generate title |

### OpenAI-compatible

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |

### Anthropic-compatible

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/messages` | Messages (streaming + non-streaming) |

## Project Structure

```
antling_api/
├── __init__.py      # Package init
├── config.py        # Configuration & env vars
├── models.py        # Pydantic models (native + OpenAI + Anthropic)
├── upstream.py      # Proxy forwarding to real server
└── server.py        # FastAPI app with all routes

client_example.py    # Demo: OpenAI SDK + Anthropic SDK usage
.env.example         # Environment template
requirements.txt     # Dependencies
pyproject.toml       # Project metadata
```

## License

MIT
