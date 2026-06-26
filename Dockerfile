# ═══════════════════════════════════════════════════════
# ant-ling Chat API — Docker image
# ═══════════════════════════════════════════════════════

# ── Stage 1: base ─────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from base stage
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY antling_api/ ./antling_api/
COPY pyproject.toml .
COPY README.md .

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run
CMD ["uvicorn", "antling_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
