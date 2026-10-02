"""
SOVEREIGN-X — RAG Chunker
Splits document text into overlapping chunks for embedding.
"""

from backend.rag.document_loader import DocumentChunk
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_file: str
    source_type: str
    document_type: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    equipment_id: Optional[str] = None
    equipment_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """
    Splits text using a hierarchy of separators with configurable chunk size and overlap.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to end at a sentence or paragraph boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    idx = text.rfind(sep, start, end)
                    if idx > start + self.chunk_size // 2:
                        end = idx + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            next_start = end - self.chunk_overlap
            # Forward-progress guard: always advance by at least 1 char
            if next_start <= start:
                next_start = start + max(1, self.chunk_size - self.chunk_overlap)
            start = next_start

            if start >= len(text):
                break

        return chunks

    def chunk_documents(
        self,
        documents: list[DocumentChunk],
        source_file: Optional[str] = None,
    ) -> list[TextChunk]:
        """Convert DocumentChunks into TextChunks."""
        all_chunks = []

        for doc in documents:
            if not doc.text or not doc.text.strip():
                continue

            splits = self.split_text(doc.text)
            for i, split_text in enumerate(splits):
                chunk_id = f"{doc.source_file}::p{doc.page_number or 0}::c{i}"
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    source_file=doc.source_file,
                    source_type=doc.source_type,
                    document_type=doc.document_type,
                    chunk_index=i,
                    total_chunks=len(splits),
                    page_number=doc.page_number,
                    section=doc.section,
                    equipment_id=doc.equipment_id,
                    equipment_type=doc.equipment_type,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(splits),
                    },
                )
                all_chunks.append(chunk)

        return all_chunks
