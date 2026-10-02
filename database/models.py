"""
SOVEREIGN-X — Database Models (SQLAlchemy)
All data stored locally in SQLite. Designed for easy PostgreSQL migration.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, JSON,
    ForeignKey, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from backend.config import get_settings
import json


class Base(DeclarativeBase):
    pass


class Mission(Base):
    __tablename__ = "missions"

    id = Column(String, primary_key=True)  # e.g. INS-2026-001
    title = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="created")  # created|running|completed|failed|pending_approval
    risk_level = Column(String, default="unknown")  # low|medium|high|critical
    classification = Column(String)  # NORMAL|WARNING|CRITICAL
    equipment_id = Column(String)
    equipment_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    recommendation = Column(Text)
    confidence_score = Column(Float, default=0.0)

    # Relationships
    files = relationship("MissionFile", back_populates="mission", cascade="all, delete-orphan")
    task_dna = relationship("TaskDNA", back_populates="mission", uselist=False, cascade="all, delete-orphan")
    audit_events = relationship("AuditEvent", back_populates="mission", cascade="all, delete-orphan")
    evidence = relationship("Evidence", back_populates="mission", cascade="all, delete-orphan")
    approvals = relationship("Approval", back_populates="mission", cascade="all, delete-orphan")
    model_calls = relationship("ModelCall", back_populates="mission", cascade="all, delete-orphan")


class MissionFile(Base):
    __tablename__ = "mission_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String)  # pdf|image|xlsx|csv|png|jpg
    file_path = Column(String)
    file_size = Column(Integer)
    source_type = Column(String, default="USER_UPLOAD")
    processed = Column(Boolean, default=False)
    ocr_applied = Column(Boolean, default=False)
    extracted_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="files")


class TaskDNA(Base):
    __tablename__ = "task_dna"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=False, unique=True)
    task_type = Column(String)
    domain = Column(String)
    equipment_type = Column(String)
    equipment_id = Column(String)
    input_types = Column(JSON, default=list)
    requires_ocr = Column(Boolean, default=False)
    requires_vision = Column(Boolean, default=False)
    requires_rag = Column(Boolean, default=True)
    requires_historical_analysis = Column(Boolean, default=False)
    requires_reasoning = Column(Boolean, default=True)
    requires_document_generation = Column(Boolean, default=False)
    requires_excel_analysis = Column(Boolean, default=False)
    risk_level = Column(String, default="low")
    selected_reasoning_model = Column(String)
    selected_vision_model = Column(String)
    selected_coding_model = Column(String)
    rag_collections = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="task_dna")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=False)
    event_type = Column(String, nullable=False)
    event_message = Column(Text)
    event_data = Column(JSON)
    severity = Column(String, default="INFO")  # INFO|WARNING|ERROR|CRITICAL
    timestamp = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="audit_events")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=False)
    filename = Column(String)
    source_type = Column(String)  # MRPL_PUBLIC|SYNTHETIC_DEMO|USER_UPLOAD
    document_type = Column(String)
    section = Column(String)
    retrieved_text = Column(Text)
    relevance_score = Column(Float, default=0.0)
    equipment_id = Column(String)
    collection_name = Column(String)
    rank = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="evidence")


class Approval(Base):
    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=False)
    recommendation = Column(Text)
    risk_level = Column(String)
    equipment_id = Column(String)
    classification = Column(String)
    confidence_score = Column(Float)
    evidence_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending|approved|modified|rejected
    approver_notes = Column(Text)
    approved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="approvals")


class ModelCall(Base):
    __tablename__ = "model_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String, ForeignKey("missions.id"), nullable=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String)  # reasoning|vision|coding|embedding
    prompt_tokens = Column(Integer, default=0)
    response_tokens = Column(Integer, default=0)
    duration_ms = Column(Integer, default=0)
    is_local = Column(Boolean, default=True)
    is_external = Column(Boolean, default=False)
    status = Column(String, default="success")
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="model_calls")


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String)  # BLOCKED_EXTERNAL|LOCAL_CALL|DATA_ACCESS|FILE_ACCESS
    source = Column(String)
    destination = Column(String)
    action = Column(String)  # ALLOWED|BLOCKED
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Equipment(Base):
    __tablename__ = "equipment"

    id = Column(String, primary_key=True)  # P-102, C-201, etc.
    equipment_type = Column(String)
    description = Column(String)
    location = Column(String)
    process_unit = Column(String)
    manufacturer = Column(String)
    model_number = Column(String)
    installation_date = Column(String)
    last_inspection = Column(DateTime)
    status = Column(String, default="ACTIVE")
    extra_data = Column(JSON, default=dict)


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(String, primary_key=True)
    equipment_id = Column(String, ForeignKey("equipment.id"))
    work_type = Column(String)
    description = Column(Text)
    priority = Column(String)
    status = Column(String, default="OPEN")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    assigned_to = Column(String)
    findings = Column(Text)
    source_type = Column(String, default="SYNTHETIC_DEMO")


# Database setup
def get_engine():
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


def create_tables(engine):
    Base.metadata.create_all(bind=engine)


def get_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
