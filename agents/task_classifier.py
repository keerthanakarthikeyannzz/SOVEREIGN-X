"""
SOVEREIGN-X — Task Classifier & Task DNA Generator
Analyzes user mission description to generate structured Task DNA.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from backend.config import get_settings

EQUIPMENT_PATTERNS = [
    (r"\bP-?(\d{3})\b", "P-{}", "Centrifugal Pump"),
    (r"\bC-?(\d{3})\b", "C-{}", "Compressor"),
    (r"\bHX-?(\d{3})\b", "HX-{}", "Heat Exchanger"),
    (r"\bPL-?(\d{3})\b", "PL-{}", "Pipeline"),
    (r"\bpump\b", None, "Centrifugal Pump"),
    (r"\bcompressor\b", None, "Compressor"),
    (r"\bheat exchanger\b", None, "Heat Exchanger"),
    (r"\bpipeline\b", None, "Pipeline"),
]

DOMAIN_KEYWORDS = {
    "refinery": ["refinery", "crude", "distillation", "CDU", "MRPL", "unit", "process"],
    "maintenance": ["maintenance", "overhaul", "repair", "service", "PM"],
    "inspection": ["inspection", "inspect", "vibration", "temperature", "bearing", "leakage", "anomaly"],
    "safety": ["safety", "LOTO", "lockout", "tagout", "permit", "hazard", "HSE"],
    "engineering": ["engineering", "calculation", "P&ID", "diagram", "design"],
}

TASK_TYPE_KEYWORDS = {
    "industrial_inspection": ["inspection", "inspect", "check", "vibration", "temperature", "leakage", "anomaly", "bearing"],
    "document_analysis": ["document", "report", "analyze", "read", "review", "PDF", "SOP"],
    "image_analysis": ["image", "photo", "photograph", "picture", "visual", "P&ID"],
    "code_generation": ["code", "python", "script", "calculate", "computation", "formula"],
    "excel_analysis": ["excel", "spreadsheet", "xlsx", "csv", "data", "table", "sensor data"],
    "historical_analysis": ["history", "historical", "trend", "previous", "past", "compare"],
    "pid_analysis": ["P&ID", "piping", "diagram", "instrumentation", "flow diagram"],
}


@dataclass
class TaskDNA:
    mission_id: str
    task_type: str = "general_reasoning"
    domain: str = "industrial"
    equipment_type: Optional[str] = None
    equipment_id: Optional[str] = None
    input_types: list = field(default_factory=list)
    requires_ocr: bool = False
    requires_vision: bool = False
    requires_rag: bool = True
    requires_historical_analysis: bool = False
    requires_reasoning: bool = True
    requires_document_generation: bool = False
    requires_excel_analysis: bool = False
    requires_code: bool = False
    risk_level: str = "low"
    selected_reasoning_model: Optional[str] = None
    selected_vision_model: Optional[str] = None
    selected_coding_model: Optional[str] = None
    rag_collections: list = field(default_factory=lambda: ["central_knowledge", "mrpl_public"])

    def to_dict(self) -> dict:
        return asdict(self)


class TaskClassifier:
    def __init__(self):
        self.settings = get_settings()

    def classify(
        self,
        mission_id: str,
        description: str,
        file_types: list[str] = None,
    ) -> TaskDNA:
        """Classify a mission and generate Task DNA."""
        desc_lower = description.lower()
        file_types = file_types or []

        dna = TaskDNA(
            mission_id=mission_id,
            selected_reasoning_model=self.settings.reasoning_model,
            selected_vision_model=self.settings.vision_model,
            selected_coding_model=self.settings.coding_model,
        )

        # Detect equipment
        equip_id, equip_type = self._detect_equipment(description)
        if equip_id:
            dna.equipment_id = equip_id
        if equip_type:
            dna.equipment_type = equip_type

        # Detect task type
        dna.task_type = self._detect_task_type(desc_lower, file_types)

        # Detect domain
        dna.domain = self._detect_domain(desc_lower)

        # Detect input types
        dna.input_types = list(set(file_types))

        # Determine required capabilities
        has_pdf = any(ft in ["pdf"] for ft in file_types)
        has_image = any(ft in ["jpg", "jpeg", "png", "image"] for ft in file_types)
        has_excel = any(ft in ["xlsx", "csv", "excel"] for ft in file_types)
        has_pid = any(ft in ["pid", "png"] for ft in file_types) or "p&id" in desc_lower

        dna.requires_ocr = has_pdf
        dna.requires_vision = has_image or has_pid
        dna.requires_excel_analysis = has_excel
        dna.requires_document_generation = any(
            kw in desc_lower for kw in ["report", "generate", "document", "approval", "note"]
        )
        dna.requires_historical_analysis = any(
            kw in desc_lower for kw in ["history", "trend", "historical", "previous", "compare"]
        ) or (equip_id is not None)  # Always check history for known equipment
        dna.requires_code = any(
            kw in desc_lower for kw in ["code", "python", "script", "calculate", "compute"]
        ) or dna.task_type == "code_generation"

        # Risk level based on task type and equipment
        dna.risk_level = self._assess_risk(desc_lower, dna)

        # RAG collections
        dna.rag_collections = self._select_rag_collections(dna)

        return dna

    def _detect_equipment(self, text: str) -> tuple[Optional[str], Optional[str]]:
        for pattern, id_fmt, eq_type in EQUIPMENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if id_fmt and match.groups():
                    eq_id = id_fmt.format(match.group(1))
                    return eq_id, eq_type
                return None, eq_type
        return None, None

    def _detect_task_type(self, desc_lower: str, file_types: list) -> str:
        # File-type based detection
        if any(ft in ["jpg", "jpeg", "png"] for ft in file_types):
            if "p&id" in desc_lower or "pid" in desc_lower:
                return "pid_analysis"
            return "image_analysis"

        scores = {}
        for task_type, keywords in TASK_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in desc_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "industrial_inspection"

        return max(scores, key=scores.get)

    def _detect_domain(self, desc_lower: str) -> str:
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in desc_lower)
            if score > 0:
                scores[domain] = score

        return max(scores, key=scores.get) if scores else "refinery"

    def _assess_risk(self, desc_lower: str, dna: TaskDNA) -> str:
        risk_score = 0

        # Critical keywords
        critical_kw = ["critical", "emergency", "failure", "shutdown", "immediate", "safety", "LOTO"]
        risk_score += sum(2 for kw in critical_kw if kw.lower() in desc_lower)

        # Warning keywords
        warning_kw = ["warning", "abnormal", "unusual", "high vibration", "leak", "overheat"]
        risk_score += sum(1 for kw in warning_kw if kw.lower() in desc_lower)

        # Equipment type risk
        if dna.equipment_type in ("Compressor", "Centrifugal Pump"):
            risk_score += 1

        if risk_score >= 4:
            return "high"
        if risk_score >= 2:
            return "medium"
        return "low"

    def _select_rag_collections(self, dna: TaskDNA) -> list:
        collections = ["central_knowledge"]
        # Always include MRPL public for context
        collections.append("mrpl_public")
        return collections
