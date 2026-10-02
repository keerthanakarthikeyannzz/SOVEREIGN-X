"""
SOVEREIGN-X — Database session management
"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
from backend.database.models import get_engine, get_session_factory, create_tables

_engine = None
_SessionFactory = None


def init_db():
    global _engine, _SessionFactory
    _engine = get_engine()
    create_tables(_engine)
    _SessionFactory = get_session_factory(_engine)
    return _engine


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for DB sessions."""
    if _SessionFactory is None:
        init_db()
    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for DB sessions (non-FastAPI use)."""
    if _SessionFactory is None:
        init_db()
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
