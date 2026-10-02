"""
SOVEREIGN-X — Model Router
Routes tasks to the appropriate local model.
All models are local Ollama models. No cloud APIs.
"""

from enum import Enum
from backend.models_layer.ollama_client import get_ollama_client, OllamaUnavailableError, ModelNotFoundError
from backend.config import get_settings


class ModelType(str, Enum):
    REASONING = "reasoning"
    VISION = "vision"
    CODING = "coding"
    EMBEDDING = "embedding"


class TaskType(str, Enum):
    INDUSTRIAL_INSPECTION = "industrial_inspection"
    DOCUMENT_ANALYSIS = "document_analysis"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    EXCEL_ANALYSIS = "excel_analysis"
    GENERAL_REASONING = "general_reasoning"
    PID_ANALYSIS = "pid_analysis"
    HISTORICAL_ANALYSIS = "historical_analysis"


TASK_TO_MODEL_TYPE = {
    TaskType.INDUSTRIAL_INSPECTION: ModelType.REASONING,
    TaskType.DOCUMENT_ANALYSIS: ModelType.REASONING,
    TaskType.IMAGE_ANALYSIS: ModelType.VISION,
    TaskType.CODE_GENERATION: ModelType.CODING,
    TaskType.EXCEL_ANALYSIS: ModelType.REASONING,
    TaskType.GENERAL_REASONING: ModelType.REASONING,
    TaskType.PID_ANALYSIS: ModelType.VISION,
    TaskType.HISTORICAL_ANALYSIS: ModelType.REASONING,
}


class ModelRouter:
    def __init__(self):
        self.settings = get_settings()
        self.client = get_ollama_client()
        self._model_status: dict[str, bool] = {}

    def get_model_for_task(self, task_type: TaskType) -> tuple[str, ModelType]:
        """Return the appropriate model name and type for a task."""
        model_type = TASK_TO_MODEL_TYPE.get(task_type, ModelType.REASONING)
        return self.get_model_by_type(model_type), model_type

    def get_model_by_type(self, model_type: ModelType) -> str:
        """Get configured model name by type."""
        mapping = {
            ModelType.REASONING: self.settings.reasoning_model,
            ModelType.VISION: self.settings.vision_model,
            ModelType.CODING: self.settings.coding_model,
            ModelType.EMBEDDING: self.settings.embedding_model,
        }
        return mapping[model_type]

    def check_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama."""
        if model_name in self._model_status:
            return self._model_status[model_name]
        available = self.client.is_model_available(model_name)
        self._model_status[model_name] = available
        return available

    def get_routing_info(self, task_dna: dict) -> dict:
        """Return routing decision for a mission's Task DNA."""
        routes = {}

        if task_dna.get("requires_reasoning") or task_dna.get("requires_rag"):
            model = self.settings.reasoning_model
            routes["reasoning"] = {
                "model": model,
                "type": ModelType.REASONING,
                "available": self.check_model_available(model),
                "task": "Document analysis, SOP comparison, recommendation generation",
            }

        if task_dna.get("requires_vision"):
            model = self.settings.vision_model
            routes["vision"] = {
                "model": model,
                "type": ModelType.VISION,
                "available": self.check_model_available(model),
                "task": "Image analysis, P&ID interpretation",
            }

        if task_dna.get("requires_code"):
            model = self.settings.coding_model
            routes["coding"] = {
                "model": model,
                "type": ModelType.CODING,
                "available": self.check_model_available(model),
                "task": "Python code generation for analysis",
            }

        embed_model = self.settings.embedding_model
        routes["embedding"] = {
            "model": embed_model,
            "type": ModelType.EMBEDDING,
            "available": self.check_model_available(embed_model),
            "task": "Document embedding for RAG retrieval",
        }

        return routes

    def get_all_model_status(self) -> dict:
        """Return status of all configured models."""
        models = {
            "reasoning": self.settings.reasoning_model,
            "vision": self.settings.vision_model,
            "coding": self.settings.coding_model,
            "embedding": self.settings.embedding_model,
        }
        result = {}
        ollama_available = self.client.is_available()
        for role, name in models.items():
            result[role] = {
                "name": name,
                "role": role,
                "ollama_running": ollama_available,
                "model_available": self.check_model_available(name) if ollama_available else False,
                "is_local": True,
                "is_external": False,
            }
        return result

    def invalidate_cache(self):
        """Clear model availability cache (call after pulling a model)."""
        self._model_status.clear()


_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
