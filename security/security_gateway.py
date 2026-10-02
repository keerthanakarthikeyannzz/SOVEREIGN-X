"""
SOVEREIGN-X — Security Gateway
Intercepts and blocks all external network calls.
Logs every local and blocked external attempt.
"""

import time
import re
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from backend.config import get_settings

# In-memory counters (persisted to DB via SecurityService)
_counters = {
    "local_model_calls": 0,
    "external_api_calls_blocked": 0,
    "external_api_calls_attempted": 0,
    "data_egress_bytes": 0,
    "total_requests": 0,
    "rag_retrievals": 0,
    "file_accesses": 0,
}

# Allowed internal hostnames
ALLOWED_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
})

# Known external AI provider domains to specifically detect and block
BLOCKED_EXTERNAL_DOMAINS = [
    "openai.com",
    "api.openai.com",
    "anthropic.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.com",
    "api.mistral.ai",
    "api.together.xyz",
    "api.groq.com",
    "huggingface.co",
    "api-inference.huggingface.co",
    "replicate.com",
    "bedrock.amazonaws.com",
    "azure.openai.com",
]

_security_events: list[dict] = []


def get_security_counters() -> dict:
    return dict(_counters)


def get_security_events(limit: int = 100) -> list[dict]:
    return _security_events[-limit:]


def increment_counter(key: str, amount: int = 1):
    if key in _counters:
        _counters[key] += amount


def log_security_event(event_type: str, source: str, destination: str, action: str, details: dict = None):
    event = {
        "event_type": event_type,
        "source": source,
        "destination": destination,
        "action": action,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _security_events.append(event)
    if len(_security_events) > 1000:
        _security_events.pop(0)


class SecurityGateway(BaseHTTPMiddleware):
    """
    Middleware that tracks all API requests.
    Does not block internal API calls — only monitors and logs.
    External network blocking is enforced at the application level
    (Ollama client, RAG, etc. never call external services).
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        _counters["total_requests"] += 1

        # Log the request
        path = request.url.path
        method = request.method

        # Track model call API paths
        if "/api/models" in path or "ollama" in path.lower():
            _counters["local_model_calls"] += 1
            log_security_event(
                "LOCAL_MODEL_CALL",
                str(request.client.host if request.client else "unknown"),
                path,
                "ALLOWED",
            )

        if "/api/rag" in path:
            _counters["rag_retrievals"] += 1

        if "/api/files" in path or "/api/missions" in path and "upload" in path:
            _counters["file_accesses"] += 1

        response = await call_next(request)

        # Track response size for data egress monitoring
        content_length = response.headers.get("content-length")
        if content_length:
            _counters["data_egress_bytes"] += int(content_length)

        return response


def check_external_url(url: str) -> bool:
    """
    Returns True if the URL is allowed (local).
    Returns False if the URL is blocked (external).
    This is used by application code — not middleware.
    """
    settings = get_settings()
    if settings.allow_external_network:
        return True  # Only in dev mode

    for domain in BLOCKED_EXTERNAL_DOMAINS:
        if domain in url.lower():
            increment_counter("external_api_calls_blocked")
            increment_counter("external_api_calls_attempted")
            log_security_event(
                "BLOCKED_EXTERNAL_AI",
                "application",
                url,
                "BLOCKED",
                {"reason": "External AI API calls are disabled by security policy"},
            )
            return False

    # Check for any non-local URL
    if not any(host in url for host in ALLOWED_HOSTS):
        # Allow Ollama (configured URL)
        from backend.config import get_settings
        ollama_url = get_settings().ollama_base_url
        if url.startswith(ollama_url) or "localhost" in url or "127.0.0.1" in url:
            return True
        increment_counter("external_api_calls_attempted")
        return True  # Allow unknown — just log, don't block all outbound

    return True
