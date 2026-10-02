"""SOVEREIGN-X — Health, Models, Security, RAG, and Code API Routers"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from backend.database.db import get_db
from backend.database import models as M
from backend.models_layer.ollama_client import get_ollama_client
from backend.models_layer.model_router import get_model_router
from backend.security.security_gateway import get_security_counters, get_security_events
from backend.rag.retriever import IntelligentRetriever
from backend.rag.vectorstore import VectorStore
from backend.config import get_settings

# ── Health ───────────────────────────────────────────────────────────────────
health_router = APIRouter(prefix="/api", tags=["health"])

@health_router.get("/health")
async def health_check():
    settings = get_settings()
    client = get_ollama_client()
    ollama_ok = client.is_available()
    vs = VectorStore()
    chroma_stats = vs.get_collection_stats()
    return {
        "status": "ok",
        "service": "SOVEREIGN-X",
        "version": "1.0.0-SIH2026",
        "ollama_available": ollama_ok,
        "external_network_allowed": settings.allow_external_network,
        "chroma_collections": chroma_stats,
        "sovereignty": "LOCAL_ONLY",
    }

# ── Models ───────────────────────────────────────────────────────────────────
models_router = APIRouter(prefix="/api/models", tags=["models"])

@models_router.get("")
async def get_models():
    router = get_model_router()
    client = get_ollama_client()
    status = router.get_all_model_status()
    installed = []
    if client.is_available():
        try:
            installed = [m["name"] for m in client.list_models()]
        except Exception:
            pass
    return {
        "models": status,
        "ollama_available": client.is_available(),
        "ollama_url": get_settings().ollama_base_url,
        "installed_models": installed,
        "external_ai_allowed": False,
    }

@models_router.post("/pull/{model_name}")
async def pull_model(model_name: str):
    """Instruction to pull a model (can't run ollama pull from API — returns instructions)."""
    return {
        "instruction": f"Run this command to download the model:",
        "command": f"ollama pull {model_name}",
        "note": "Model download runs locally. No data leaves the machine.",
    }

# ── Security ─────────────────────────────────────────────────────────────────
security_router = APIRouter(prefix="/api/security", tags=["security"])

@security_router.get("/status")
async def get_security_status(db: Session = Depends(get_db)):
    counters = get_security_counters()
    settings = get_settings()
    client = get_ollama_client()

    # Count model calls from DB
    total_local_calls = db.query(M.ModelCall).filter(M.ModelCall.is_local == True).count()
    total_external = db.query(M.ModelCall).filter(M.ModelCall.is_external == True).count()

    return {
        "network_status": "OFFLINE" if not settings.allow_external_network else "MONITORED",
        "external_ai_calls": total_external,
        "external_api_requests": counters.get("external_api_calls_blocked", 0),
        "data_egress_bytes": counters.get("data_egress_bytes", 0),
        "local_model_calls": total_local_calls,
        "blocked_external_attempts": counters.get("external_api_calls_blocked", 0),
        "total_requests": counters.get("total_requests", 0),
        "rag_retrievals": counters.get("rag_retrievals", 0),
        "ollama_available": client.is_available(),
        "allow_external_network": settings.allow_external_network,
        "sovereignty_status": "SECURE" if not settings.allow_external_network else "PERMISSIVE",
    }

@security_router.get("/events")
async def get_security_events_api():
    events = get_security_events(limit=100)
    return {"events": events, "count": len(events)}

# ── RAG ──────────────────────────────────────────────────────────────────────
rag_router = APIRouter(prefix="/api/rag", tags=["rag"])

class RAGSearchRequest(BaseModel):
    query: str
    equipment_id: Optional[str] = None
    equipment_type: Optional[str] = None
    mission_id: Optional[str] = None
    n_results: int = 5

@rag_router.post("/search")
async def rag_search(payload: RAGSearchRequest):
    retriever = IntelligentRetriever()
    if not retriever.embedder.is_available():
        return {
            "results": [],
            "error": f"Embedding model '{get_settings().embedding_model}' not available. Run: ollama pull {get_settings().embedding_model}",
        }
    results = retriever.retrieve(
        query=payload.query,
        equipment_id=payload.equipment_id,
        equipment_type=payload.equipment_type,
        mission_id=payload.mission_id,
        n_results=payload.n_results,
    )
    return {
        "results": [
            {
                "text": r["text"][:500],
                "filename": r["metadata"].get("source_file", ""),
                "source_type": r["metadata"].get("source_type", ""),
                "document_type": r["metadata"].get("document_type", ""),
                "score": r["score"],
                "rank": r["rank"],
            }
            for r in results
        ],
        "count": len(results),
        "query": payload.query,
    }

@rag_router.get("/stats")
async def rag_stats():
    vs = VectorStore()
    return {"collections": vs.get_collection_stats()}

@rag_router.post("/index")
async def trigger_indexing(background_tasks):
    """Trigger background indexing of the knowledge base."""
    def do_index():
        from backend.rag.indexer import run_indexer
        run_indexer(verbose=False)

    background_tasks.add_task(do_index)
    return {"message": "Indexing started in background"}

# ── Code ─────────────────────────────────────────────────────────────────────
code_router = APIRouter(prefix="/api/code", tags=["code"])

class CodeGenerateRequest(BaseModel):
    task: str
    context: Optional[str] = ""
    mission_id: Optional[str] = None

@code_router.post("/generate")
async def generate_code(payload: CodeGenerateRequest):
    settings = get_settings()
    client = get_ollama_client()
    if not client.is_available():
        return {"error": "Ollama not available", "code": None}
    if not client.is_model_available(settings.coding_model):
        return {"error": f"Coding model {settings.coding_model} not installed", "code": None}

    prompt = f"""Write Python code to accomplish this industrial data analysis task:

Task: {payload.task}

Context: {payload.context or 'Industrial sensor data analysis'}

Requirements:
- Use pandas for data manipulation
- Print clear results
- Handle errors gracefully
- Do not access external URLs or APIs

Return ONLY the Python code, no explanations."""

    result = client.generate(
        model=settings.coding_model,
        prompt=prompt,
        system="You are an industrial data analysis code generator. Write clean, safe Python code.",
        temperature=0.1,
    )

    # Extract code block
    code = result["response"]
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    return {
        "code": code,
        "model": result["model"],
        "is_local": True,
    }

@code_router.post("/execute")
async def execute_code(payload: dict):
    """Execute code in sandbox. Requires Docker."""
    import subprocess
    import shutil

    if not shutil.which("docker"):
        return {
            "executed": False,
            "error": "Docker is not installed. Secure code execution requires Docker.",
            "instruction": "Install Docker Desktop from https://docker.com to enable sandboxed execution.",
            "security_note": "Code execution on the host machine is disabled for security. Docker provides network isolation, CPU/memory limits, and filesystem restrictions.",
        }

    return {
        "executed": False,
        "error": "Docker sandbox not yet implemented in this deployment.",
        "code": payload.get("code", ""),
    }
