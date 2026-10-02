"""
SOVEREIGN-X — Ollama Local AI Client
Wraps Ollama HTTP API. NEVER calls external AI APIs.
If Ollama is unavailable, returns a clear error — never falls back to cloud.
"""

import httpx
import base64
import json
import time
from pathlib import Path
from typing import Optional, AsyncIterator
from backend.config import get_settings


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not reachable."""
    pass


class ModelNotFoundError(Exception):
    """Raised when a requested model is not installed in Ollama."""
    pass


class OllamaClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self._timeout = httpx.Timeout(300.0, connect=10.0)

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self._timeout)

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            with self._client() as c:
                r = c.get("/api/tags", timeout=5.0)
                return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        """Return list of installed models."""
        try:
            with self._client() as c:
                r = c.get("/api/tags")
                r.raise_for_status()
                return r.json().get("models", [])
        except httpx.ConnectError:
            raise OllamaUnavailableError(
                "Ollama is not running. Start it with: ollama serve"
            )

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is installed."""
        try:
            models = self.list_models()
            installed = [m["name"] for m in models]
            # Support both "llama3.2:3b" and "llama3.2" style
            for installed_name in installed:
                if model_name in installed_name or installed_name in model_name:
                    return True
            return False
        except OllamaUnavailableError:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate text using local Ollama model."""
        if not self.is_available():
            raise OllamaUnavailableError(
                "Ollama is not running. Start it with: ollama serve"
            )
        if not self.is_model_available(model):
            raise ModelNotFoundError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        start = time.time()
        with self._client() as c:
            r = c.post("/api/generate", json=payload)
            r.raise_for_status()
            result = r.json()

        duration_ms = int((time.time() - start) * 1000)
        return {
            "response": result.get("response", ""),
            "model": model,
            "duration_ms": duration_ms,
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "response_tokens": result.get("eval_count", 0),
            "is_local": True,
        }

    def generate_with_image(
        self,
        model: str,
        prompt: str,
        image_path: str,
        system: Optional[str] = None,
    ) -> dict:
        """Generate text using vision model with an image."""
        if not self.is_available():
            raise OllamaUnavailableError("Ollama is not running.")
        if not self.is_model_available(model):
            raise ModelNotFoundError(f"Vision model '{model}' is not installed. Run: ollama pull {model}")

        # Read and encode image
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if system:
            payload["system"] = system

        start = time.time()
        with self._client() as c:
            r = c.post("/api/generate", json=payload)
            r.raise_for_status()
            result = r.json()

        duration_ms = int((time.time() - start) * 1000)
        return {
            "response": result.get("response", ""),
            "model": model,
            "duration_ms": duration_ms,
            "is_local": True,
        }

    def embed(self, model: str, text: str) -> list[float]:
        """Generate embeddings using local embedding model."""
        if not self.is_available():
            raise OllamaUnavailableError("Ollama is not running.")

        with self._client() as c:
            r = c.post("/api/embed", json={"model": model, "input": text})
            r.raise_for_status()
            result = r.json()
            embeddings = result.get("embeddings", result.get("embedding", []))
            if isinstance(embeddings, list) and len(embeddings) > 0:
                if isinstance(embeddings[0], list):
                    return embeddings[0]
                return embeddings
            return []

    def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch API call."""
        if not texts:
            return []
        if not self.is_available():
            raise OllamaUnavailableError("Ollama is not running.")

        with self._client() as c:
            r = c.post("/api/embed", json={"model": model, "input": texts})
            r.raise_for_status()
            result = r.json()
            embeddings = result.get("embeddings", [])
            if isinstance(embeddings, list):
                return embeddings
            return [[] for _ in texts]

    def chat(
        self,
        model: str,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        """Chat completion using local model."""
        if not self.is_available():
            raise OllamaUnavailableError("Ollama is not running.")
        if not self.is_model_available(model):
            raise ModelNotFoundError(f"Model '{model}' is not installed.")

        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        start = time.time()
        with self._client() as c:
            r = c.post("/api/chat", json=payload)
            r.raise_for_status()
            result = r.json()

        return {
            "response": result.get("message", {}).get("content", ""),
            "model": model,
            "duration_ms": int((time.time() - start) * 1000),
            "is_local": True,
        }


# Singleton
_client_instance: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance
