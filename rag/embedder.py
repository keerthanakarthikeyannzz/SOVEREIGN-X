"""
SOVEREIGN-X — RAG Embedder
Generates embeddings using nomic-embed-text via Ollama.
Completely local — no external API.
"""

from backend.models_layer.ollama_client import get_ollama_client, OllamaUnavailableError
from backend.config import get_settings
from backend.rag.chunker import TextChunk
from typing import Optional
import time


class LocalEmbedder:
    def __init__(self):
        self.settings = get_settings()
        self.client = get_ollama_client()
        self.model = self.settings.embedding_model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        text = text.strip()
        if not text:
            return []
        return self.client.embed(self.model, text)

    def embed_chunks(
        self,
        chunks: list[TextChunk],
        progress_callback=None,
    ) -> list[list[float]]:
        """Embed all chunks, return list of embedding vectors."""
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        try:
            embeddings = self.client.embed_batch(self.model, texts)
            if progress_callback:
                progress_callback(len(chunks), len(chunks))
            return embeddings
        except Exception:
            # Fallback to sequential if batch fails
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    emb = self.embed_text(chunk.text)
                    embeddings.append(emb)
                except Exception:
                    embeddings.append([])
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
            return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        return self.embed_text(query)

    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self.client.is_available() and self.client.is_model_available(self.model)
