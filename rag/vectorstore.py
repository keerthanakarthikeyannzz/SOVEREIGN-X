"""
SOVEREIGN-X — ChromaDB Vector Store
Three collections: central_knowledge, mrpl_public, user_cases
All embeddings are local (nomic-embed-text via Ollama).
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import Optional
from backend.config import get_settings
from backend.rag.chunker import TextChunk

COLLECTION_CENTRAL = "central_knowledge"
COLLECTION_MRPL = "mrpl_public"
COLLECTION_USER = "user_cases"

ALL_COLLECTIONS = [COLLECTION_CENTRAL, COLLECTION_MRPL, COLLECTION_USER]

_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings = get_settings()
        chroma_path = Path(settings.chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_or_create_collection(name: str) -> chromadb.Collection:
    global _client
    client = get_chroma_client()
    try:
        return client.get_or_create_collection(
            name=name,
            metadata={"description": f"SOVEREIGN-X collection: {name}"},
        )
    except Exception:
        # Re-initialize client if collection metadata error occurs
        _client = None
        client = get_chroma_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"description": f"SOVEREIGN-X collection: {name}"},
        )


class VectorStore:
    def __init__(self):
        self.settings = get_settings()

    def _collection_for_source(self, source_type: str, mission_id: Optional[str] = None) -> str:
        if source_type == "MRPL_PUBLIC":
            return COLLECTION_MRPL
        if source_type == "USER_UPLOAD":
            return COLLECTION_USER
        return COLLECTION_CENTRAL

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]):
        """Add text chunks with their embeddings to the appropriate collection."""
        try:
            # Group by collection
            by_collection: dict[str, list[tuple[TextChunk, list[float]]]] = {}
            for chunk, emb in zip(chunks, embeddings):
                col = self._collection_for_source(chunk.source_type)
                if col not in by_collection:
                    by_collection[col] = []
                by_collection[col].append((chunk, emb))

            for col_name, items in by_collection.items():
                collection = get_or_create_collection(col_name)
                ids = []
                embeds = []
                documents = []
                metadatas = []

                for chunk, emb in items:
                    ids.append(chunk.chunk_id)
                    embeds.append(emb)
                    documents.append(chunk.text)
                    metadatas.append({
                        "source_file": chunk.source_file,
                        "source_type": chunk.source_type,
                        "document_type": chunk.document_type,
                        "page_number": str(chunk.page_number or ""),
                        "equipment_id": chunk.equipment_id or "",
                        "equipment_type": chunk.equipment_type or "",
                        "chunk_index": chunk.chunk_index,
                        "section": chunk.section or "",
                    })

                # Upsert in batches of 100
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    collection.upsert(
                        ids=ids[i:i+batch_size],
                        embeddings=embeds[i:i+batch_size],
                        documents=documents[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size],
                    )
        except Exception as e:
            print(f"VectorStore add_chunks error (continuing mission): {e}")

    def search(
        self,
        query_embedding: list[float],
        collection_names: Optional[list[str]] = None,
        n_results: int = 6,
        equipment_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> list[dict]:
        """Search across specified collections and return ranked results."""
        if collection_names is None:
            collection_names = [COLLECTION_CENTRAL, COLLECTION_MRPL]

        # Include user cases if mission_id provided
        if mission_id and COLLECTION_USER not in collection_names:
            collection_names = list(collection_names) + [COLLECTION_USER]

        all_results = []

        for col_name in collection_names:
            try:
                collection = get_or_create_collection(col_name)

                # Build where filter for equipment-aware retrieval
                where = None
                if equipment_id:
                    # Try equipment_id filter (if equipment_id is set in metadata)
                    # ChromaDB 'where' requires exact match — we'll post-filter
                    pass

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, max(1, collection.count())),
                    include=["documents", "metadatas", "distances"],
                )

                if results and results["ids"] and results["ids"][0]:
                    for idx, (doc_id, doc, meta, dist) in enumerate(zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )):
                        # Convert distance to similarity score (cosine)
                        score = max(0.0, 1.0 - dist)

                        # Boost score if equipment matches
                        if equipment_id and meta.get("equipment_id") == equipment_id:
                            score = min(1.0, score * 1.3)
                        if equipment_type and equipment_type.lower() in meta.get("equipment_type", "").lower():
                            score = min(1.0, score * 1.15)

                        # Filter out user cases from other missions
                        if col_name == COLLECTION_USER and mission_id:
                            chunk_mission = meta.get("mission_id", "")
                            if chunk_mission and chunk_mission != mission_id:
                                continue

                        all_results.append({
                            "id": doc_id,
                            "text": doc,
                            "metadata": meta,
                            "score": score,
                            "collection": col_name,
                            "rank": 0,
                        })
            except Exception as e:
                # Collection may be empty — that's fine
                continue

        # Sort by score descending and re-rank
        all_results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(all_results[:n_results]):
            r["rank"] = i + 1

        return all_results[:n_results]

    def get_collection_stats(self) -> dict:
        stats = {}
        for col_name in ALL_COLLECTIONS:
            try:
                col = get_or_create_collection(col_name)
                stats[col_name] = {"count": col.count()}
            except Exception:
                stats[col_name] = {"count": 0, "error": True}
        return stats

    def delete_mission_docs(self, mission_id: str):
        """Remove all user case documents for a specific mission."""
        try:
            col = get_or_create_collection(COLLECTION_USER)
            # Get IDs with matching mission_id
            results = col.get(where={"mission_id": mission_id})
            if results and results["ids"]:
                col.delete(ids=results["ids"])
        except Exception:
            pass
