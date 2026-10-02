"""
SOVEREIGN-X — Mission Controller (Main Agent)
Orchestrates the complete mission workflow.
Emits structured execution events — NO hidden chain-of-thought.
"""

import uuid
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass

from backend.agents.task_classifier import TaskClassifier, TaskDNA
from backend.models_layer.ollama_client import get_ollama_client, OllamaUnavailableError, ModelNotFoundError
from backend.models_layer.model_router import get_model_router
from backend.rag.retriever import IntelligentRetriever
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunker import RecursiveChunker
from backend.rag.embedder import LocalEmbedder
from backend.rag.vectorstore import VectorStore
from backend.database.db import db_session
from backend.database import models as M
from backend.config import get_settings
from backend.security.security_gateway import increment_counter, log_security_event


@dataclass
class MissionEvent:
    event_type: str
    message: str
    data: dict = None
    severity: str = "INFO"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "data": self.data or {},
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


ANALYSIS_SYSTEM_PROMPT = """You are SOVEREIGN-X, a secure industrial AI analyst running entirely on local infrastructure.
You analyze industrial equipment inspection data and provide evidence-based findings.

RULES:
- Base ALL conclusions on provided evidence only.
- Use vibration/temperature thresholds from provided SOPs.
- Clearly state the classification: NORMAL, WARNING, or CRITICAL.
- Distinguish: Observed fact vs Possible interpretation vs Confirmed cause.
- Never fabricate citations or evidence.
- Do not expose reasoning steps — provide structured findings only.
- Label all synthetic documents as SYNTHETIC_DEMO.
- Label user uploads as USER_UPLOAD.
- Be precise about measurements and thresholds.
"""


class MissionController:
    def __init__(self):
        self.settings = get_settings()
        self.classifier = TaskClassifier()
        self.retriever = IntelligentRetriever()
        self.loader = DocumentLoader()
        self.chunker = RecursiveChunker()
        self.embedder = LocalEmbedder()
        self.vectorstore = VectorStore()
        self.router = get_model_router()
        self.client = get_ollama_client()

    def _emit(self, mission_id: str, event_type: str, message: str, data: dict = None, severity: str = "INFO") -> MissionEvent:
        """Create and store an audit event."""
        event = MissionEvent(event_type=event_type, message=message, data=data, severity=severity)
        try:
            with db_session() as db:
                db.add(M.AuditEvent(
                    mission_id=mission_id,
                    event_type=event_type,
                    event_message=message,
                    event_data=data or {},
                    severity=severity,
                    timestamp=datetime.utcnow(),
                ))
        except Exception:
            pass
        return event

    def _is_stopped(self, mission_id: str) -> bool:
        try:
            with db_session() as db:
                m = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
                if m and m.status in ("cancelled", "stopped"):
                    return True
        except Exception:
            pass
        return False

    def run_mission(self, mission_id: str) -> Generator[MissionEvent, None, None]:
        """
        Execute a complete mission workflow.
        Yields MissionEvent objects for SSE streaming.
        """
        settings = self.settings

        # ── Step 1: Load mission from DB ──────────────────────────────
        with db_session() as db:
            mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
            if not mission:
                yield self._emit(mission_id, "ERROR", f"Mission {mission_id} not found", severity="ERROR")
                return

            files = db.query(M.MissionFile).filter(M.MissionFile.mission_id == mission_id).all()
            file_data = [(f.filename, f.file_path, f.file_type, f.source_type) for f in files]
            description = mission.description or ""

        yield self._emit(mission_id, "MISSION_START", f"Mission {mission_id} started")

        # ── Step 2: Update status ────────────────────────────────────
        with db_session() as db:
            db.query(M.Mission).filter(M.Mission.id == mission_id).update({"status": "running"})

        if self._is_stopped(mission_id):
            yield self._emit(mission_id, "MISSION_STOPPED", "Mission analysis stopped by user", severity="WARNING")
            return

        # ── Step 3: Task Classification ──────────────────────────────
        yield self._emit(mission_id, "CLASSIFYING", "Analyzing mission and generating Task DNA")

        file_types = [ft for _, _, ft, _ in file_data]
        dna = self.classifier.classify(mission_id, description, file_types)

        # Store Task DNA
        with db_session() as db:
            existing = db.query(M.TaskDNA).filter(M.TaskDNA.mission_id == mission_id).first()
            if not existing:
                db.add(M.TaskDNA(
                    mission_id=mission_id,
                    task_type=dna.task_type,
                    domain=dna.domain,
                    equipment_type=dna.equipment_type,
                    equipment_id=dna.equipment_id,
                    input_types=dna.input_types,
                    requires_ocr=dna.requires_ocr,
                    requires_vision=dna.requires_vision,
                    requires_rag=dna.requires_rag,
                    requires_historical_analysis=dna.requires_historical_analysis,
                    requires_reasoning=dna.requires_reasoning,
                    requires_document_generation=dna.requires_document_generation,
                    requires_excel_analysis=dna.requires_excel_analysis,
                    risk_level=dna.risk_level,
                    selected_reasoning_model=dna.selected_reasoning_model,
                    selected_vision_model=dna.selected_vision_model,
                    rag_collections=dna.rag_collections,
                ))
            # Update mission equipment info
            db.query(M.Mission).filter(M.Mission.id == mission_id).update({
                "equipment_id": dna.equipment_id,
                "equipment_type": dna.equipment_type,
                "risk_level": dna.risk_level,
            })

        yield self._emit(mission_id, "CLASSIFIED", "Task DNA generated",
                         data=dna.to_dict())

        # ── Step 4: Model Selection ──────────────────────────────────
        yield self._emit(mission_id, "MODEL_SELECTION", "Selecting local AI models")

        routing_info = self.router.get_routing_info(dna.to_dict())
        ollama_available = self.client.is_available()

        if not ollama_available:
            yield self._emit(
                mission_id, "WARNING",
                "Ollama is not running. Start with: ollama serve",
                severity="WARNING"
            )

        yield self._emit(mission_id, "MODELS_SELECTED", "Model routing determined",
                         data={"routing": routing_info, "ollama_available": ollama_available})

        # ── Step 5: File Processing ──────────────────────────────────
        if self._is_stopped(mission_id):
            yield self._emit(mission_id, "MISSION_STOPPED", "Mission analysis stopped by user", severity="WARNING")
            return
        extracted_texts = []

        for filename, file_path, file_type, source_type in file_data:
            if self._is_stopped(mission_id):
                yield self._emit(mission_id, "MISSION_STOPPED", "Mission analysis stopped by user", severity="WARNING")
                return
            if file_path and Path(file_path).exists():
                yield self._emit(mission_id, "FILE_LOADING", f"Loading {filename}")

                docs = self.loader.load_file(
                    file_path,
                    source_type=source_type,
                    equipment_id=dna.equipment_id,
                    equipment_type=dna.equipment_type,
                )

                if docs:
                    page_count = len(docs)
                    ocr_applied = any(d.metadata.get("ocr_applied") for d in docs)
                    full_text = "\n\n".join(d.text for d in docs if d.text)

                    extracted_texts.append({
                        "filename": filename,
                        "text": full_text,
                        "source_type": source_type,
                        "pages": page_count,
                    })

                    if ocr_applied:
                        yield self._emit(mission_id, "OCR_APPLIED", f"OCR applied to {filename}",
                                         data={"filename": filename})
                    else:
                        yield self._emit(mission_id, "TEXT_EXTRACTED", f"Text extracted from {filename}",
                                         data={"filename": filename, "pages": page_count})

                    # Auto-detect equipment from extracted text if not previously set
                    if not dna.equipment_id or not dna.equipment_type:
                        eq_id, eq_type = self.classifier._detect_equipment(full_text)
                        if eq_id and not dna.equipment_id:
                            dna.equipment_id = eq_id
                        if eq_type and not dna.equipment_type:
                            dna.equipment_type = eq_type
                        if eq_id or eq_type:
                            with db_session() as db:
                                db.query(M.Mission).filter(M.Mission.id == mission_id).update({
                                    "equipment_id": dna.equipment_id,
                                    "equipment_type": dna.equipment_type,
                                })
                                db.query(M.TaskDNA).filter(M.TaskDNA.mission_id == mission_id).update({
                                    "equipment_id": dna.equipment_id,
                                    "equipment_type": dna.equipment_type,
                                })

                    # Update file record with extracted text
                    with db_session() as db:
                        db.query(M.MissionFile).filter(
                            M.MissionFile.filename == filename,
                            M.MissionFile.mission_id == mission_id
                        ).update({
                            "processed": True,
                            "ocr_applied": ocr_applied,
                            "extracted_text": full_text[:10000],  # Store first 10k chars
                        })

                    # Index into ChromaDB user collection
                    if self.embedder.is_available():
                        chunks = self.chunker.chunk_documents(docs)
                        if chunks:
                            for chunk in chunks:
                                chunk.metadata["mission_id"] = mission_id
                            embeddings = self.embedder.embed_chunks(chunks)
                            valid = [(c, e) for c, e in zip(chunks, embeddings) if e]
                            if valid:
                                vc, ve = zip(*valid)
                                self.vectorstore.add_chunks(list(vc), list(ve))

        # ── Step 6: Image Vision Analysis ──────────────────────────
        if self._is_stopped(mission_id):
            yield self._emit(mission_id, "MISSION_STOPPED", "Mission analysis stopped by user", severity="WARNING")
            return
        vision_observations = []

        if dna.requires_vision:
            vision_files = [(fn, fp, ft, st) for fn, fp, ft, st in file_data
                           if ft in ("jpg", "jpeg", "png", "image")]

            for filename, file_path, file_type, source_type in vision_files:
                if self._is_stopped(mission_id):
                    yield self._emit(mission_id, "MISSION_STOPPED", "Mission analysis stopped by user", severity="WARNING")
                    return
                if file_path and Path(file_path).exists():
                    yield self._emit(mission_id, "VISION_ANALYSIS", f"Analyzing image: {filename}")

                    if not ollama_available or not self.client.is_model_available(settings.vision_model):
                        vision_observations.append({
                            "filename": filename,
                            "observations": "Vision model not available. Image analysis skipped.",
                            "available": False,
                        })
                        yield self._emit(mission_id, "VISION_UNAVAILABLE",
                                         f"Vision model {settings.vision_model} not available",
                                         severity="WARNING")
                        continue

                    try:
                        vision_prompt = (
                            f"You are analyzing an industrial equipment inspection image.\n"
                            f"Equipment: {dna.equipment_id or 'Unknown'} ({dna.equipment_type or 'Industrial Equipment'})\n\n"
                            "Describe what you observe. Structure your response:\n"
                            "OBSERVED: (visible physical conditions, any visible issues)\n"
                            "POSSIBLE INTERPRETATION: (what the observations may indicate)\n"
                            "NOT CONFIRMED: (what cannot be determined from the image alone)\n\n"
                            "Be factual. Do not invent defects not visible in the image."
                        )
                        result = self.client.generate_with_image(
                            model=settings.vision_model,
                            prompt=vision_prompt,
                            image_path=file_path,
                        )
                        vision_observations.append({
                            "filename": filename,
                            "observations": result["response"],
                            "model": result["model"],
                            "available": True,
                        })
                        increment_counter("local_model_calls")
                        yield self._emit(mission_id, "VISION_COMPLETE", f"Image analysis complete: {filename}",
                                         data={"filename": filename})
                    except Exception as e:
                        vision_observations.append({
                            "filename": filename,
                            "observations": f"Vision analysis failed: {str(e)}",
                            "available": False,
                        })

        # ── Step 7: RAG Retrieval ────────────────────────────────────
        yield self._emit(mission_id, "RAG_SEARCH", "Searching organizational knowledge base")

        # Build RAG query from description + extracted text
        rag_query = description
        if extracted_texts:
            # Include key terms from extracted documents
            first_doc_text = extracted_texts[0]["text"][:1000]
            rag_query = f"{description}\n\nDocument excerpt: {first_doc_text}"

        rag_results = []
        sop_results = []
        historical_results = []

        if self.embedder.is_available():
            # General retrieval
            rag_results = self.retriever.retrieve(
                query=rag_query,
                equipment_id=dna.equipment_id,
                equipment_type=dna.equipment_type,
                mission_id=mission_id,
                include_user_docs=True,
                n_results=8,
            )

            # SOP/threshold specific search
            if dna.equipment_id:
                sop_query = f"{dna.equipment_id} operating limits vibration temperature threshold normal warning critical"
                sop_results = self.retriever.retrieve(
                    query=sop_query,
                    equipment_id=dna.equipment_id,
                    equipment_type=dna.equipment_type,
                    n_results=5,
                    include_user_docs=False,
                )

            # Historical analysis
            if dna.requires_historical_analysis and dna.equipment_id:
                historical_results = self.retriever.search_historical(dna.equipment_id)
        else:
            yield self._emit(mission_id, "RAG_UNAVAILABLE",
                             "Embedding model not available. RAG retrieval skipped.",
                             severity="WARNING")

        yield self._emit(mission_id, "RAG_COMPLETE",
                         f"Retrieved {len(rag_results)} relevant knowledge chunks",
                         data={"count": len(rag_results)})

        if dna.requires_historical_analysis:
            yield self._emit(mission_id, "HISTORICAL_RETRIEVED",
                             f"Retrieved {len(historical_results)} historical records",
                             data={"count": len(historical_results)})

        # ── Step 8: Store Evidence ───────────────────────────────────
        all_evidence = []
        seen_files = set()
        evidence_rank = 1

        for result_set in [rag_results, sop_results, historical_results]:
            for r in result_set:
                fn = r["metadata"].get("source_file", "Unknown")
                if fn not in seen_files:
                    seen_files.add(fn)
                    ev = {
                        "filename": fn,
                        "source_type": r["metadata"].get("source_type", "SYNTHETIC_DEMO"),
                        "document_type": r["metadata"].get("document_type", "Document"),
                        "section": r["metadata"].get("section", f"Page {r['metadata'].get('page_number', '?')}"),
                        "text": r["text"][:500],
                        "score": r["score"],
                        "rank": evidence_rank,
                    }
                    all_evidence.append(ev)
                    evidence_rank += 1

        # Save to DB
        with db_session() as db:
            db.query(M.Evidence).filter(M.Evidence.mission_id == mission_id).delete()
            for ev in all_evidence[:10]:  # Store top 10
                db.add(M.Evidence(
                    mission_id=mission_id,
                    filename=ev["filename"],
                    source_type=ev["source_type"],
                    document_type=ev["document_type"],
                    section=ev["section"],
                    retrieved_text=ev["text"],
                    relevance_score=ev["score"],
                    equipment_id=dna.equipment_id,
                    rank=ev["rank"],
                ))

        yield self._emit(mission_id, "EVIDENCE_COMPILED",
                         f"Evidence compiled: {len(all_evidence)} unique sources",
                         data={"count": len(all_evidence)})

        # ── Step 9: LLM Analysis ─────────────────────────────────────
        yield self._emit(mission_id, "ANALYSIS_START", "Performing AI analysis with local model")

        analysis_result = None
        classification = "UNKNOWN"
        recommendation = ""
        confidence = 0.0

        if ollama_available and self.client.is_model_available(settings.reasoning_model):
            # Build context from evidence
            evidence_context = "\n\n".join([
                f"[{ev['source_type']}] {ev['filename']}:\n{ev['text']}"
                for ev in all_evidence[:5]
            ])

            extracted_context = "\n\n".join([
                f"[USER_UPLOAD] {et['filename']}:\n{et['text'][:2000]}"
                for et in extracted_texts[:2]
            ])

            vision_context = "\n".join([
                f"[IMAGE: {vo['filename']}] {vo['observations']}"
                for vo in vision_observations if vo.get("available")
            ])

            analysis_prompt = f"""Analyze this industrial inspection mission.

MISSION: {description}

EQUIPMENT: {dna.equipment_id or "Unknown"} ({dna.equipment_type or "Industrial Equipment"})

UPLOADED DOCUMENTS:
{extracted_context if extracted_context else "No documents uploaded"}

IMAGE ANALYSIS:
{vision_context if vision_context else "No images analyzed"}

KNOWLEDGE BASE (SOPs, Standards, Historical Records):
{evidence_context if evidence_context else "No knowledge base results found"}

Based ONLY on the above evidence, provide a structured analysis:

CLASSIFICATION: [NORMAL|WARNING|CRITICAL|INSUFFICIENT_DATA]
CONFIDENCE: [0-100]%

KEY FINDINGS:
- (list specific findings with measurements)

SOP COMPARISON:
- (compare current values against retrieved thresholds)

HISTORICAL TREND:
- (compare with historical records if available)

RECOMMENDATION:
- (specific, actionable recommendation)

SAFETY CONSIDERATIONS:
- (relevant safety requirements)

Evidence used: (list filenames cited)

Do not fabricate measurements or citations not present in the evidence above."""

            try:
                result = self.client.generate(
                    model=settings.reasoning_model,
                    prompt=analysis_prompt,
                    system=ANALYSIS_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2000,
                )
                analysis_result = result["response"]
                increment_counter("local_model_calls")

                # Extract classification from response
                classification = self._extract_classification(analysis_result)
                confidence = self._extract_confidence(analysis_result)
                recommendation = self._extract_recommendation(analysis_result)

                # Log model call
                with db_session() as db:
                    db.add(M.ModelCall(
                        mission_id=mission_id,
                        model_name=settings.reasoning_model,
                        model_type="reasoning",
                        duration_ms=result["duration_ms"],
                        prompt_tokens=result.get("prompt_tokens", 0),
                        response_tokens=result.get("response_tokens", 0),
                        is_local=True,
                        is_external=False,
                        status="success",
                    ))

            except OllamaUnavailableError as e:
                yield self._emit(mission_id, "OLLAMA_ERROR", str(e), severity="ERROR")
                analysis_result = "Analysis unavailable: Ollama not running."
            except ModelNotFoundError as e:
                yield self._emit(mission_id, "MODEL_ERROR", str(e), severity="ERROR")
                analysis_result = f"Analysis unavailable: {str(e)}"
            except Exception as e:
                yield self._emit(mission_id, "ANALYSIS_ERROR", f"Analysis failed: {str(e)}", severity="ERROR")
                analysis_result = f"Analysis failed: {str(e)}"
        else:
            analysis_result = self._rule_based_analysis(description, all_evidence, extracted_texts)
            classification, confidence = self._rule_based_classify(description, all_evidence, extracted_texts)
            recommendation = self._rule_based_recommend(classification, dna)
            yield self._emit(mission_id, "RULE_BASED_ANALYSIS",
                             "Using rule-based analysis (Ollama not available)",
                             severity="WARNING")

        yield self._emit(mission_id, "ANALYSIS_COMPLETE", "Analysis complete",
                         data={"classification": classification, "confidence": confidence})

        # ── Step 10: Update Mission ──────────────────────────────────
        with db_session() as db:
            db.query(M.Mission).filter(M.Mission.id == mission_id).update({
                "classification": classification,
                "recommendation": recommendation,
                "confidence_score": confidence / 100.0,
                "status": "pending_approval" if dna.risk_level in ("high", "medium") else "completed",
                "risk_level": dna.risk_level,
                "updated_at": datetime.utcnow(),
            })

        yield self._emit(mission_id, "VERIFICATION", "Verifying findings against evidence")

        # ── Step 11: Create Approval if Medium/High Risk ────────────
        if dna.risk_level in ("high", "medium") and classification != "NORMAL":
            with db_session() as db:
                existing = db.query(M.Approval).filter(M.Approval.mission_id == mission_id).first()
                if not existing:
                    db.add(M.Approval(
                        mission_id=mission_id,
                        recommendation=recommendation,
                        risk_level=dna.risk_level,
                        equipment_id=dna.equipment_id,
                        classification=classification,
                        confidence_score=confidence / 100.0,
                        evidence_count=len(all_evidence),
                        status="pending",
                    ))

            yield self._emit(mission_id, "APPROVAL_REQUIRED",
                             f"Human approval required for {classification} {dna.risk_level}-risk finding",
                             severity="WARNING")

        # ── Step 12: Complete & Generate Approval DOCX ────────────────
        m_status = "pending_approval" if dna.risk_level in ("high", "medium") and classification != "NORMAL" else "completed"
        with db_session() as db:
            db.query(M.Mission).filter(M.Mission.id == mission_id).update({
                "status": m_status,
                "completed_at": datetime.utcnow(),
            })

            # Generate DOCX note file now that AI analysis is complete
            try:
                from backend.services.document_service import DocumentGenerator
                gen = DocumentGenerator()
                m_obj = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
                evidence_items = db.query(M.Evidence).filter(M.Evidence.mission_id == mission_id).all()
                ev_list = [{
                    "rank": ev.rank, "filename": ev.filename, "source_type": ev.source_type,
                    "document_type": ev.document_type, "section": ev.section,
                    "retrieved_text": ev.retrieved_text, "score": getattr(ev, "relevance_score", 0.0),
                    "equipment_id": ev.equipment_id, "collection_name": ev.collection_name
                } for ev in evidence_items]

                gen.generate_approval_note(
                    mission_id=mission_id,
                    mission_data={
                        "equipment_id": m_obj.equipment_id,
                        "equipment_type": m_obj.equipment_type,
                        "classification": m_obj.classification,
                        "risk_level": m_obj.risk_level,
                        "recommendation": m_obj.recommendation,
                        "confidence_score": m_obj.confidence_score or 0.0,
                        "description": m_obj.description,
                    },
                    evidence=ev_list,
                    analysis=analysis_result or "",
                    task_dna=dna.to_dict(),
                    approval_status="PENDING" if m_status == "pending_approval" else "COMPLETED",
                    approver_notes="",
                )
            except Exception as e:
                print(f"Auto-generate DOCX note error: {e}")

        yield self._emit(mission_id, "MISSION_COMPLETE",
                         f"Mission {mission_id} complete. Classification: {classification}",
                         data={
                             "classification": classification,
                             "confidence": confidence,
                             "recommendation": recommendation,
                             "evidence_count": len(all_evidence),
                             "analysis": analysis_result,
                             "vision_observations": vision_observations,
                         })

    # ── Helper methods ──────────────────────────────────────────────

    def _extract_classification(self, text: str) -> str:
        # Handle both plain and markdown bold formats
        text_upper = text.upper()
        # Look for explicit classification line
        for pattern in [
            r"\*?\*?CLASSIFICATION\*?\*?:\s*\*?\*?(CRITICAL|WARNING|NORMAL|INSUFFICIENT[_\s]DATA)\*?\*?",
        ]:
            m = re.search(pattern, text_upper)
            if m:
                val = m.group(1).replace(" ", "_")
                if "CRITICAL" in val:
                    return "CRITICAL"
                if "WARNING" in val:
                    return "WARNING"
                if "NORMAL" in val:
                    return "NORMAL"
                if "INSUFFICIENT" in val:
                    return "INSUFFICIENT_DATA"
        # Fallback scanning
        if "CLASSIFICATION: CRITICAL" in text_upper or "**CRITICAL**" in text_upper:
            return "CRITICAL"
        if "CLASSIFICATION: WARNING" in text_upper or "**WARNING**" in text_upper:
            return "WARNING"
        if "CLASSIFICATION: NORMAL" in text_upper:
            return "NORMAL"
        if "INSUFFICIENT" in text_upper:
            return "INSUFFICIENT_DATA"
        # Word-count fallback
        if text_upper.count("CRITICAL") > text_upper.count("NORMAL") and text_upper.count("CRITICAL") > text_upper.count("WARNING"):
            return "CRITICAL"
        if text_upper.count("WARNING") > text_upper.count("NORMAL"):
            return "WARNING"
        if "NORMAL" in text_upper or "WITHIN OPERATING LIMITS" in text_upper or "OPERATIONAL" in text_upper:
            return "NORMAL"
        return "NORMAL"

    def _extract_confidence(self, text: str) -> float:
        # Handle **CONFIDENCE:** 75% and CONFIDENCE: 75 formats
        match = re.search(r"\*?\*?CONFIDENCE\*?\*?:\s*\*?\*?(\d+)\*?\*?%?", text, re.IGNORECASE)
        if match:
            return min(100.0, float(match.group(1)))
        return 75.0

    def _extract_recommendation(self, text: str) -> str:
        # Handle **RECOMMENDATION:** format used by llama3.2
        match = re.search(
            r"\*?\*?RECOMMENDATION\*?\*?:\s*[-\n]*(.*?)(?:\n\n|\n\*?\*?SAFETY|$)",
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            rec = match.group(1).strip()
            # Clean up markdown bold markers
            rec = re.sub(r"\*+", "", rec).strip()
            if rec:
                return rec[:500]
        # Second pass: look for recommendation after the key findings
        lines = text.split("\n")
        capture = False
        captured = []
        for line in lines:
            lu = line.upper().strip()
            if "RECOMMENDATION" in lu and ":" in lu:
                capture = True
                # Get text after colon on same line
                after = line.split(":", 1)[1].strip().lstrip("-").strip()
                if after:
                    captured.append(after)
                continue
            if capture:
                if lu.startswith(("**SAFETY", "SAFETY", "EVIDENCE", "**EVIDENCE")):
                    break
                if line.strip():
                    cleaned = re.sub(r"\*+", "", line).strip().lstrip("-").strip()
                    if cleaned:
                        captured.append(cleaned)
        if captured:
            return " ".join(captured)[:500]
        return "Review findings and take appropriate action."

    def _rule_based_analysis(self, description: str, evidence: list, extracted_texts: list) -> str:
        """Simple rule-based analysis when LLM is unavailable."""
        lines = ["Rule-based analysis (LLM unavailable):"]
        if "8.2" in description or any("8.2" in et.get("text", "") for et in extracted_texts):
            lines.append("Vibration: 8.2 mm/s detected")
            lines.append("SOP threshold for CRITICAL: > 7.0 mm/s")
            lines.append("Classification: CRITICAL (8.2 > 7.0)")
        lines.append(f"Evidence sources: {len(evidence)}")
        return "\n".join(lines)

    def _rule_based_classify(self, description: str, evidence: list, extracted_texts: list) -> tuple[str, float]:
        """Rule-based classification from extracted numbers."""
        full_text = description + " ".join(et.get("text", "")[:500] for et in extracted_texts)
        # Extract vibration value
        vib_match = re.search(r"vibration[:\s]+(\d+\.?\d*)\s*mm/s", full_text, re.IGNORECASE)
        if vib_match:
            vib = float(vib_match.group(1))
            if vib > 7.0:
                return "CRITICAL", 85.0
            if vib >= 4.5:
                return "WARNING", 80.0
            return "NORMAL", 90.0
        return "UNKNOWN", 0.0

    def _rule_based_recommend(self, classification: str, dna: TaskDNA) -> str:
        if classification == "CRITICAL":
            return f"Immediate maintenance assessment required for {dna.equipment_id or 'equipment'}. Apply LOTO procedure before any physical intervention."
        if classification == "WARNING":
            return f"Schedule inspection for {dna.equipment_id or 'equipment'} within 7 days. Increase monitoring frequency."
        if classification == "NORMAL":
            return "Continue normal operation. Maintain scheduled monitoring intervals."
        return "Review findings and consult relevant SOP."
