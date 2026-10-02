"""
SOVEREIGN-X — Missions API Router
"""

import uuid
import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.db import get_db, init_db, db_session
from backend.database import models as M
from backend.agents.mission_controller import MissionController
from backend.services.document_service import DocumentGenerator
from backend.config import get_settings

router = APIRouter(prefix="/api/missions", tags=["missions"])
settings = get_settings()


class MissionCreate(BaseModel):
    title: Optional[str] = None
    description: str


class ApprovalAction(BaseModel):
    action: str  # approve | modify | reject
    notes: Optional[str] = ""


def generate_mission_id() -> str:
    now = datetime.utcnow()
    return f"INS-{now.year}-{str(uuid.uuid4())[:4].upper()}"


def mission_to_dict(m: M.Mission) -> dict:
    return {
        "id": m.id,
        "title": m.title,
        "description": m.description,
        "status": m.status,
        "risk_level": m.risk_level,
        "classification": m.classification,
        "equipment_id": m.equipment_id,
        "equipment_type": m.equipment_type,
        "recommendation": m.recommendation,
        "confidence_score": m.confidence_score,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        "completed_at": m.completed_at.isoformat() if m.completed_at else None,
    }


@router.post("")
async def create_mission(payload: MissionCreate, db: Session = Depends(get_db)):
    """Create a new mission."""
    mission_id = generate_mission_id()
    title = payload.title or f"Mission {mission_id}"

    mission = M.Mission(
        id=mission_id,
        title=title,
        description=payload.description,
        status="created",
    )
    db.add(mission)
    db.commit()
    db.refresh(mission)

    return {"mission": mission_to_dict(mission), "message": "Mission created"}


@router.get("")
async def list_missions(db: Session = Depends(get_db)):
    """List all missions."""
    missions = db.query(M.Mission).order_by(M.Mission.created_at.desc()).all()
    return {"missions": [mission_to_dict(m) for m in missions]}


@router.get("/{mission_id}")
async def get_mission(mission_id: str, db: Session = Depends(get_db)):
    """Get a single mission with full details."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    result = mission_to_dict(mission)

    # Task DNA
    dna = db.query(M.TaskDNA).filter(M.TaskDNA.mission_id == mission_id).first()
    if dna:
        result["task_dna"] = {
            "task_type": dna.task_type,
            "domain": dna.domain,
            "equipment_type": dna.equipment_type,
            "equipment_id": dna.equipment_id,
            "input_types": dna.input_types or [],
            "requires_ocr": dna.requires_ocr,
            "requires_vision": dna.requires_vision,
            "requires_rag": dna.requires_rag,
            "requires_historical_analysis": dna.requires_historical_analysis,
            "requires_reasoning": dna.requires_reasoning,
            "requires_document_generation": dna.requires_document_generation,
            "risk_level": dna.risk_level,
            "selected_reasoning_model": dna.selected_reasoning_model,
            "selected_vision_model": dna.selected_vision_model,
            "rag_collections": dna.rag_collections or [],
        }

    # Files
    files = db.query(M.MissionFile).filter(M.MissionFile.mission_id == mission_id).all()
    result["files"] = [
        {
            "filename": f.filename,
            "file_type": f.file_type,
            "file_size": f.file_size,
            "source_type": f.source_type,
            "processed": f.processed,
            "ocr_applied": f.ocr_applied,
        }
        for f in files
    ]

    # Approval
    approval = db.query(M.Approval).filter(M.Approval.mission_id == mission_id).first()
    if approval:
        result["approval"] = {
            "status": approval.status,
            "recommendation": approval.recommendation,
            "risk_level": approval.risk_level,
            "classification": approval.classification,
            "confidence_score": approval.confidence_score,
            "evidence_count": approval.evidence_count,
            "approver_notes": approval.approver_notes,
            "approved_at": approval.approved_at.isoformat() if approval.approved_at else None,
        }

    return result


@router.delete("/{mission_id}")
async def delete_mission(mission_id: str, db: Session = Depends(get_db)):
    """Delete a mission report and all associated resources."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    # Try vectorstore cleanup if available
    try:
        from backend.rag.vectorstore import VectorStore
        vs = VectorStore()
        vs.delete_mission_docs(mission_id)
    except Exception as e:
        print(f"Vectorstore cleanup warning for {mission_id}: {e}")

    # Remove uploaded case files directory
    case_dir = Path(settings.data_dir) / "current_cases" / mission_id
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)

    # Delete mission from DB (cascade deletes related records)
    db.delete(mission)
    db.commit()

    return {"message": "Mission deleted successfully", "mission_id": mission_id}

async def upload_files(
    mission_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Upload files to a mission."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    # Create case directory
    case_dir = Path(settings.data_dir) / "current_cases" / mission_id
    case_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for upload_file in files:
        filename = upload_file.filename
        file_path = case_dir / filename
        ext = Path(filename).suffix.lower().lstrip(".")

        # Detect file type
        image_exts = {"jpg", "jpeg", "png"}
        excel_exts = {"xlsx", "xls", "csv"}
        if ext == "pdf":
            file_type = "pdf"
        elif ext in image_exts:
            file_type = "image"
        elif ext in excel_exts:
            file_type = ext
        else:
            file_type = ext

        # Save file
        content = await upload_file.read()
        if len(content) > settings.max_upload_size:
            raise HTTPException(status_code=413, detail=f"File {filename} exceeds max upload size")

        with open(file_path, "wb") as f:
            f.write(content)

        # Store in DB
        mission_file = M.MissionFile(
            mission_id=mission_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            file_size=len(content),
            source_type="USER_UPLOAD",
        )
        db.add(mission_file)
        uploaded.append({"filename": filename, "file_type": file_type, "size": len(content)})

    db.commit()
    return {"uploaded": uploaded, "count": len(uploaded)}


def _run_mission_background(mission_id: str):
    """Run mission analysis in background thread with error handling."""
    try:
        controller = MissionController()
        events = list(controller.run_mission(mission_id))
    except Exception as e:
        print(f"Error executing mission {mission_id}: {e}")
        try:
            with db_session() as db:
                db.query(M.Mission).filter(M.Mission.id == mission_id).update({
                    "status": "failed",
                    "recommendation": f"Mission failed: {str(e)}"
                })
        except Exception:
            pass


@router.post("/{mission_id}/analyze")
async def analyze_mission(mission_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Trigger mission analysis."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    if mission.status == "running":
        raise HTTPException(status_code=400, detail="Mission is already running")

    background_tasks.add_task(_run_mission_background, mission_id)
    return {"message": "Analysis started", "mission_id": mission_id}


@router.post("/{mission_id}/stop")
async def stop_mission(mission_id: str, db: Session = Depends(get_db)):
    """Stop/cancel an ongoing mission analysis process."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission.status = "cancelled"
    db.add(M.AuditEvent(
        mission_id=mission_id,
        event_type="MISSION_STOPPED",
        event_message=f"Mission {mission_id} analysis stopped by user",
        event_data={"stopped_by": "user"},
        severity="WARNING",
        timestamp=datetime.utcnow()
    ))
    db.commit()
    return {"message": "Mission analysis stopped", "mission_id": mission_id, "status": "cancelled"}


@router.get("/{mission_id}/events")
async def get_mission_events(mission_id: str, db: Session = Depends(get_db)):
    """Get audit events for a mission (for SSE polling)."""
    events = (
        db.query(M.AuditEvent)
        .filter(M.AuditEvent.mission_id == mission_id)
        .order_by(M.AuditEvent.timestamp.asc())
        .all()
    )
    return {
        "events": [
            {
                "event_type": e.event_type,
                "message": e.event_message,
                "data": e.event_data or {},
                "severity": e.severity,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in events
        ]
    }


@router.get("/{mission_id}/stream")
async def stream_mission_events(mission_id: str, db: Session = Depends(get_db)):
    """SSE stream for real-time mission events from audit log."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    async def event_generator():
        import json
        last_id = 0
        while True:
            with db_session() as session:
                events = (
                    session.query(M.AuditEvent)
                    .filter(M.AuditEvent.mission_id == mission_id, M.AuditEvent.id > last_id)
                    .order_by(M.AuditEvent.id.asc())
                    .all()
                )
                for e in events:
                    last_id = e.id
                    event_dict = {
                        "event_type": e.event_type,
                        "message": e.event_message,
                        "data": e.event_data or {},
                        "severity": e.severity,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    }
                    yield f"data: {json.dumps(event_dict)}\n\n"
                    if e.event_type in ("MISSION_COMPLETE", "ERROR"):
                        yield "data: {\"event_type\": \"STREAM_END\"}\n\n"
                        return

                m = session.query(M.Mission).filter(M.Mission.id == mission_id).first()
                if m and m.status not in ("created", "running"):
                    yield "data: {\"event_type\": \"STREAM_END\"}\n\n"
                    return

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{mission_id}/evidence")
async def get_evidence(mission_id: str, db: Session = Depends(get_db)):
    """Get all evidence for a mission."""
    evidence = (
        db.query(M.Evidence)
        .filter(M.Evidence.mission_id == mission_id)
        .order_by(M.Evidence.rank.asc())
        .all()
    )
    return {
        "evidence": [
            {
                "id": e.id,
                "filename": e.filename,
                "source_type": e.source_type,
                "document_type": e.document_type,
                "section": e.section,
                "text": e.retrieved_text,
                "score": e.relevance_score,
                "rank": e.rank,
                "equipment_id": e.equipment_id,
            }
            for e in evidence
        ]
    }


@router.post("/{mission_id}/approve")
async def approve_mission(
    mission_id: str,
    payload: ApprovalAction,
    db: Session = Depends(get_db),
):
    """Approve, modify, or reject a mission recommendation."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    approval = db.query(M.Approval).filter(M.Approval.mission_id == mission_id).first()

    action = payload.action.lower()
    status_map = {"approve": "approved", "modify": "modified", "reject": "rejected"}
    new_status = status_map.get(action, "pending")

    if approval:
        approval.status = new_status
        approval.approver_notes = payload.notes or ""
        approval.approved_at = datetime.utcnow()
    else:
        db.add(M.Approval(
            mission_id=mission_id,
            recommendation=mission.recommendation or "",
            status=new_status,
            approver_notes=payload.notes or "",
            approved_at=datetime.utcnow(),
            classification=mission.classification,
            risk_level=mission.risk_level,
            confidence_score=mission.confidence_score,
        ))

    mission.status = "completed"
    db.add(M.AuditEvent(
        mission_id=mission_id,
        event_type="APPROVAL_ACTION",
        event_message=f"Mission {action}d by operator",
        event_data={"action": action, "notes": payload.notes},
        severity="INFO",
    ))
    db.commit()

    return {"message": f"Mission {action}d", "status": new_status}


@router.post("/{mission_id}/generate-document")
async def generate_document(mission_id: str, db: Session = Depends(get_db)):
    """Generate approval_note.docx for a mission."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    if mission.status == "created":
        raise HTTPException(
            status_code=400,
            detail="Mission not analyzed yet. Trigger /analyze first."
        )

    evidence = (
        db.query(M.Evidence)
        .filter(M.Evidence.mission_id == mission_id)
        .order_by(M.Evidence.rank.asc())
        .all()
    )
    dna = db.query(M.TaskDNA).filter(M.TaskDNA.mission_id == mission_id).first()
    approval = db.query(M.Approval).filter(M.Approval.mission_id == mission_id).first()

    # Get latest analysis from audit events
    analysis_event = (
        db.query(M.AuditEvent)
        .filter(
            M.AuditEvent.mission_id == mission_id,
            M.AuditEvent.event_type == "MISSION_COMPLETE",
        )
        .order_by(M.AuditEvent.timestamp.desc())
        .first()
    )
    analysis_text = ""
    if analysis_event and analysis_event.event_data:
        analysis_text = analysis_event.event_data.get("analysis", "")

    gen = DocumentGenerator()
    file_path = gen.generate_approval_note(
        mission_id=mission_id,
        mission_data={
            "equipment_id": mission.equipment_id,
            "equipment_type": mission.equipment_type,
            "classification": mission.classification,
            "risk_level": mission.risk_level,
            "recommendation": mission.recommendation,
            "confidence_score": mission.confidence_score or 0,
            "description": mission.description,
        },
        evidence=[
            {
                "filename": e.filename,
                "source_type": e.source_type,
                "document_type": e.document_type,
                "section": e.section,
                "score": e.relevance_score,
                "rank": e.rank,
            }
            for e in evidence
        ],
        analysis=analysis_text,
        task_dna={
            "task_type": dna.task_type if dna else None,
            "domain": dna.domain if dna else None,
            "equipment_type": dna.equipment_type if dna else None,
            "requires_ocr": dna.requires_ocr if dna else False,
            "requires_vision": dna.requires_vision if dna else False,
            "requires_rag": dna.requires_rag if dna else True,
            "risk_level": dna.risk_level if dna else "unknown",
            "selected_reasoning_model": dna.selected_reasoning_model if dna else None,
        } if dna else None,
        approval_status=approval.status.upper() if approval else "PENDING",
        approver_notes=approval.approver_notes if approval else "",
    )

    db.add(M.AuditEvent(
        mission_id=mission_id,
        event_type="DOCUMENT_GENERATED",
        event_message="Approval note generated",
        event_data={"file_path": file_path},
    ))
    db.commit()

    return FileResponse(
        path=file_path,
        filename=f"{mission_id}_approval_note.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.post("/{mission_id}/demo-load")
async def load_demo_case(
    mission_id: str,
    case: str = "case_001_pump",
    db: Session = Depends(get_db),
):
    """Load a demo case into a mission (for SIH Demo Mode)."""
    mission = db.query(M.Mission).filter(M.Mission.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")

    # Resolve demo dir — supports both flat and nested DEMO_CASES/ structure
    data_path = Path(settings.data_dir)
    demo_dir = None
    for candidate in [
        data_path / "demo_cases" / case,
        data_path / "demo_cases" / "DEMO_CASES" / case,
    ]:
        if candidate.exists():
            demo_dir = candidate
            break

    if not demo_dir:
        raise HTTPException(status_code=404, detail=f"Demo case '{case}' not found in demo_cases/")

    # Create case directory for mission
    case_dir = data_path / "current_cases" / mission_id
    case_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for demo_file in demo_dir.glob("*.*"):
        dest = case_dir / demo_file.name
        shutil.copy2(demo_file, dest)
        ext = demo_file.suffix.lower().lstrip(".")
        file_type = "pdf" if ext == "pdf" else ("image" if ext in ("jpg", "jpeg", "png") else ext)

        existing = db.query(M.MissionFile).filter(
            M.MissionFile.mission_id == mission_id,
            M.MissionFile.filename == demo_file.name
        ).first()
        if not existing:
            db.add(M.MissionFile(
                mission_id=mission_id,
                filename=demo_file.name,
                file_type=file_type,
                file_path=str(dest),
                file_size=demo_file.stat().st_size,
                source_type="USER_UPLOAD",
            ))
        uploaded.append(demo_file.name)

    # Attach relevant inspection image based on case type
    img_dir = data_path / "images"
    image_map = {
        "pump": "P102_bearing_area_inspection.jpg",
        "compressor": "C201_compressor_inspection.jpg",
        "heat_exchanger": "HX301_heat_exchanger_inspection.jpg",
    }
    for keyword, img_name in image_map.items():
        if keyword in case.lower():
            img_file = img_dir / img_name
            if img_file.exists():
                dest = case_dir / img_file.name
                shutil.copy2(img_file, dest)
                existing = db.query(M.MissionFile).filter(
                    M.MissionFile.mission_id == mission_id,
                    M.MissionFile.filename == img_file.name
                ).first()
                if not existing:
                    db.add(M.MissionFile(
                        mission_id=mission_id,
                        filename=img_file.name,
                        file_type="image",
                        file_path=str(dest),
                        file_size=img_file.stat().st_size,
                        source_type="USER_UPLOAD",
                    ))
                uploaded.append(img_file.name)
            break

    db.commit()
    return {"loaded": uploaded, "count": len(uploaded), "demo_case": case}

