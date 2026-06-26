"""
Configuration for ant-ling API.

Authentication
--------------
The real chat.ant-ling.com API uses TWO auth mechanisms:

1. **Cookie-based session** (primary):
   - ``TLingSESSIONID`` — JWT session token issued after login.
   - ``jsh_t_c_e`` — tracking/analytics cookie.
   - ``spanner`` — infrastructure routing cookie.

2. **Header-based tenant identification** (required on every request):
   - ``tenant-id`` — identifies the tenant/organization.

No ``Authorization: Bearer`` header is used by the native API.

Environment variables
---------------------
ANTLING_BASE_URL
    Upstream server URL (e.g. https://chat.ant-ling.com).
    If unset, the server runs in standalone/mock mode.

ANTLING_TENANT_ID
    Tenant ID header value (default: 20260523LTJY01501627).

ANTLING_SESSION_COOKIE
    Full cookie string containing TLingSESSIONID and other cookies.
    Copy this from your browser's DevTools > Network tab.

ANTLING_API_KEY
    Optional Bearer token for local proxy authentication (not used
    by the real upstream API).

OPENAI_API_KEY / ANTHROPIC_API_KEY
    Keys for the OpenAI/Anthropic adapter layers.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── Upstream ────────────────────────────────────
BASE_URL = os.getenv("ANTLING_BASE_URL", "").strip()
TENANT_ID = os.getenv("ANTLING_TENANT_ID", "20260523LTJY01501627")
API_KEY = os.getenv("ANTLING_API_KEY", "sk-no-key-required")

# ── Cookie auth ─────────────────────────────────
# Full cookie string from browser. Example:
# "TLingSESSIONID=eyJhbGciOiJIUzI1NiIs...; jsh_t_c_e=...; spanner=..."
SESSION_COOKIE = os.getenv("ANTLING_SESSION_COOKIE", "").strip()

# ── Server ──────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Models ──────────────────────────────────────
MODELS = ["Ring-2.6-1T", "Ling-2.6-1T", "Ling-2.6-flash", "Ling-2.5-1T", "Ring-2.5-1T", "Ming-Omni-flash", "AntAngelMed"]

# ── Model cache ─────────────────────────────────
# TTL for the model list cache in standalone mode (seconds).
# In proxy mode the list is always fetched from upstream (no caching).
MODEL_CACHE_TTL = 3600  # 1 hour

# ── Adapter keys ────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-no-key-required")


def is_proxy_mode() -> bool:
    """Check if proxy mode is enabled (forwarding to real upstream)."""
    return bool(BASE_URL)
