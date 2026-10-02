"""
SOVEREIGN-X — Intelligent RAG Retriever
Equipment-aware retrieval that routes queries to relevant collections.
P-102 queries get pump docs. C-201 queries get compressor docs.
"""

from typing import Optional
from backend.rag.embedder import LocalEmbedder
from backend.rag.vectorstore import VectorStore, COLLECTION_CENTRAL, COLLECTION_MRPL, COLLECTION_USER
from backend.config import get_settings

EQUIPMENT_TYPE_KEYWORDS = {
    "centrifugal_pump": ["pump", "centrifugal", "P-10", "P-11", "vibration", "seal", "bearing"],
    "compressor": ["compressor", "C-20", "discharge", "suction", "rpm"],
    "heat_exchanger": ["heat exchanger", "HX", "fouling", "tube", "shell", "delta T"],
    "pipeline": ["pipeline", "pipe", "line", "pig", "corrosion", "thickness"],
}

EQUIPMENT_PREFIX_MAP = {
    "P-": "centrifugal_pump",
    "C-": "compressor",
    "HX-": "heat_exchanger",
    "PL-": "pipeline",
}


def detect_equipment_type_from_id(equipment_id: str) -> Optional[str]:
    if not equipment_id:
        return None
    for prefix, eq_type in EQUIPMENT_PREFIX_MAP.items():
        if equipment_id.upper().startswith(prefix):
            return eq_type
    return None


def detect_equipment_type_from_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    for eq_type, keywords in EQUIPMENT_TYPE_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            return eq_type
    return None


class IntelligentRetriever:
    def __init__(self):
        self.embedder = LocalEmbedder()
        self.vectorstore = VectorStore()

    def retrieve(
        self,
        query: str,
        equipment_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
        mission_id: Optional[str] = None,
        include_user_docs: bool = True,
        n_results: int = 6,
    ) -> list[dict]:
        """
        Equipment-aware RAG retrieval.
        Automatically detects equipment type from query/ID.
        Prioritizes relevant collections.
        """
        if not self.embedder.is_available():
            return self._fallback_results(query)

        # Auto-detect equipment type if not provided
        if not equipment_type and equipment_id:
            equipment_type = detect_equipment_type_from_id(equipment_id)
        if not equipment_type:
            equipment_type = detect_equipment_type_from_text(query)

        # Embed query
        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            return []

        # Determine collections to search
        collections = [COLLECTION_CENTRAL, COLLECTION_MRPL]
        if include_user_docs and mission_id:
            collections.append(COLLECTION_USER)

        # Perform search
        results = self.vectorstore.search(
            query_embedding=query_embedding,
            collection_names=collections,
            n_results=n_results,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            mission_id=mission_id,
        )

        # Post-filter: if equipment_type detected, prefer relevant docs
        if equipment_type:
            results = self._boost_relevant_docs(results, equipment_type)

        return results

    def _boost_relevant_docs(self, results: list[dict], equipment_type: str) -> list[dict]:
        """Boost relevance scores for documents matching equipment type."""
        keywords = EQUIPMENT_TYPE_KEYWORDS.get(equipment_type, [])
        if not keywords:
            return results

        for r in results:
            text_lower = r["text"].lower()
            filename_lower = r["metadata"].get("source_file", "").lower()
            # Check for keyword matches
            matches = sum(1 for kw in keywords if kw.lower() in text_lower or kw.lower() in filename_lower)
            if matches > 0:
                r["score"] = min(1.0, r["score"] * (1 + 0.1 * matches))

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def _fallback_results(self, query: str) -> list[dict]:
        """Return empty results when embedder unavailable."""
        return []

    def search_for_sop_comparison(
        self,
        equipment_id: str,
        metric: str,
        value: float,
    ) -> list[dict]:
        """
        Specialized search to find SOP/standard thresholds for a given metric.
        Example: equipment_id=P-102, metric=vibration, value=8.2
        """
        equipment_type = detect_equipment_type_from_id(equipment_id)
        query = (
            f"{equipment_id} {equipment_type} {metric} threshold limit "
            f"normal warning critical classification operating limits SOP"
        )
        return self.retrieve(
            query=query,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            n_results=8,
            include_user_docs=False,
        )

    def search_historical(self, equipment_id: str) -> list[dict]:
        """Search for historical records for a specific equipment."""
        equipment_type = detect_equipment_type_from_id(equipment_id)
        query = f"{equipment_id} historical record maintenance vibration temperature trend"
        return self.retrieve(
            query=query,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            n_results=5,
            include_user_docs=False,
        )
