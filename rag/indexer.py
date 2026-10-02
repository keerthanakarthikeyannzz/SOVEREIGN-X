"""
SOVEREIGN-X — RAG Indexer
Indexes all documents from the central knowledge base and MRPL public knowledge.
Run this once to populate the vector database.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.rag.document_loader import DocumentLoader
from backend.rag.chunker import RecursiveChunker
from backend.rag.embedder import LocalEmbedder
from backend.rag.vectorstore import VectorStore
from backend.config import get_settings

settings = get_settings()

DATA_DIR = Path(settings.data_dir) if not Path(settings.data_dir).is_absolute() else Path(settings.data_dir)
if not DATA_DIR.is_absolute():
    DATA_DIR = project_root / DATA_DIR

# The dataset has two possible directory structures:
# 1. data/central_knowledge/SOP/  (flat - synthetic scripts wrote here)
# 2. data/central_knowledge/CENTRAL_KNOWLEDGE_BASE/SOP/  (nested - from dataset import)
# We resolve to whichever actually exists, collecting both.

KB_FLAT = DATA_DIR / "central_knowledge"
KB_NESTED = DATA_DIR / "central_knowledge" / "CENTRAL_KNOWLEDGE_BASE"


def _dir_entries(subdir: str, doc_type: str, source_type: str = "SYNTHETIC_DEMO") -> list[dict]:
    """Return indexing entries for both flat and nested paths if they exist."""
    entries = []
    for base in [KB_FLAT, KB_NESTED]:
        p = base / subdir
        if p.exists():
            entries.append({
                "dir": p,
                "collection": "central_knowledge",
                "source_type": source_type,
                "document_type": doc_type,
            })
    return entries


# Document collections to index (resolves real paths automatically)
INDEXING_MAP = (
    _dir_entries("SOP", "SOP") +
    _dir_entries("Standards", "Operating Limits") +
    _dir_entries("Historical", "Historical Record") +
    _dir_entries("Equipment", "Equipment Profile") +
    _dir_entries("Engineering", "Engineering Reference") +
    # Also index mrpl_public if it exists
    ([{
        "dir": DATA_DIR / "mrpl_public",
        "collection": "mrpl_public",
        "source_type": "MRPL_PUBLIC",
        "document_type": "MRPL Document",
    }] if (DATA_DIR / "mrpl_public").exists() else [])
)

EQUIPMENT_ID_MAP = {
    "P102": {"equipment_id": "P-102", "equipment_type": "Centrifugal Pump"},
    "P103": {"equipment_id": "P-103", "equipment_type": "Centrifugal Pump"},
    "C201": {"equipment_id": "C-201", "equipment_type": "Compressor"},
    "HX301": {"equipment_id": "HX-301", "equipment_type": "Heat Exchanger"},
    "pump": {"equipment_id": None, "equipment_type": "Centrifugal Pump"},
    "compressor": {"equipment_id": None, "equipment_type": "Compressor"},
    "heat_exchanger": {"equipment_id": None, "equipment_type": "Heat Exchanger"},
    "pipeline": {"equipment_id": None, "equipment_type": "Pipeline"},
}


def detect_equipment(filename: str) -> dict:
    fn = filename.upper().replace("_", "").replace("-", "")
    for key, meta in EQUIPMENT_ID_MAP.items():
        if key.upper().replace("_", "") in fn:
            return meta
    return {"equipment_id": None, "equipment_type": None}


def run_indexer(verbose: bool = True):
    loader = DocumentLoader()
    chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
    embedder = LocalEmbedder()
    vectorstore = VectorStore()

    if not embedder.is_available():
        print("ERROR: Embedding model not available.")
        print(f"  Run: ollama pull {settings.embedding_model}")
        print("  Then: ollama serve")
        return False

    total_chunks = 0
    total_files = 0

    for entry in INDEXING_MAP:
        doc_dir = Path(entry["dir"])
        if not doc_dir.exists():
            if verbose:
                print(f"  SKIP (not found): {doc_dir}")
            continue

        files = list(doc_dir.glob("**/*.pdf")) + list(doc_dir.glob("**/*.txt")) + list(doc_dir.glob("**/*.csv"))

        for file_path in files:
            if verbose:
                print(f"  Indexing: {file_path.name} [{entry['source_type']}]")

            equip = detect_equipment(file_path.name)

            # Load document
            docs = loader.load_file(
                str(file_path),
                source_type=entry["source_type"],
                equipment_id=equip.get("equipment_id"),
                equipment_type=equip.get("equipment_type"),
            )

            if not docs:
                continue

            # Override document type
            for doc in docs:
                doc.document_type = entry.get("document_type", doc.document_type)

            # Chunk
            chunks = chunker.chunk_documents(docs)
            if not chunks:
                continue

            if verbose:
                print(f"    -> {len(chunks)} chunks")

            # Embed
            texts = [c.text for c in chunks]
            embeddings = []
            for i, text in enumerate(texts):
                emb = embedder.embed_text(text)
                embeddings.append(emb)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    -> Embedded {i+1}/{len(texts)}")

            # Filter out empty embeddings
            valid = [(c, e) for c, e in zip(chunks, embeddings) if e]
            if not valid:
                continue

            valid_chunks, valid_embeddings = zip(*valid)

            # Store in ChromaDB
            vectorstore.add_chunks(list(valid_chunks), list(valid_embeddings))

            total_chunks += len(valid_chunks)
            total_files += 1

    stats = vectorstore.get_collection_stats()
    if verbose:
        print(f"\nIndexing complete!")
        print(f"  Files processed: {total_files}")
        print(f"  Chunks indexed: {total_chunks}")
        print(f"  Collection stats: {stats}")

    return True


if __name__ == "__main__":
    print("SOVEREIGN-X — RAG Indexer")
    print("=" * 50)
    print(f"Data directory: {DATA_DIR}")
    print()

    success = run_indexer(verbose=True)
    if not success:
        sys.exit(1)
    print("\nDone. Vector database is ready.")
