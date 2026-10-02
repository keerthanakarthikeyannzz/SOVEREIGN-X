"""
SOVEREIGN-X — FastAPI Main Application
"""

import sys
import os
from pathlib import Path

# Ensure backend package is importable from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env from project root
from dotenv import load_dotenv
env_path = project_root / ".env"
if not env_path.exists():
    # Copy from .env.example if .env doesn't exist
    example_path = project_root / ".env.example"
    if example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.config import get_settings
from backend.database.db import init_db
from backend.security.security_gateway import SecurityGateway
from backend.api.missions import router as missions_router
from backend.api.routers import (
    health_router, models_router, security_router,
    rag_router, code_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and resources on startup."""
    settings = get_settings()

    # Init database
    init_db()

    # Create required directories
    for d in [settings.data_dir, settings.generated_dir, settings.logs_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"SOVEREIGN-X backend started")
    print(f"  Ollama URL: {settings.ollama_base_url}")
    print(f"  External network: {'ALLOWED' if settings.allow_external_network else 'BLOCKED'}")
    print(f"  Reasoning model: {settings.reasoning_model}")
    print(f"  Vision model: {settings.vision_model}")
    print(f"  Embedding model: {settings.embedding_model}")

    yield

    print("SOVEREIGN-X backend shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="SOVEREIGN-X API",
        description="Sovereign On-Premise Agentic AI Workbench — Smart India Hackathon 2026",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security gateway middleware
    app.add_middleware(SecurityGateway)

    # Routers
    app.include_router(missions_router)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(security_router)
    app.include_router(rag_router)
    app.include_router(code_router)

    @app.get("/")
    async def root():
        return {
            "service": "SOVEREIGN-X",
            "version": "1.0.0-SIH2026",
            "description": "Sovereign On-Premise Agentic AI Workbench",
            "organization": "MRPL",
            "hackathon": "Smart India Hackathon 2026",
            "external_ai": False,
            "local_only": True,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
