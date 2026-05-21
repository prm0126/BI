"""Database engine + session factory.

Picks SQLite at `./bi_forecast.db` by default. Override with the
`DATABASE_URL` env var, e.g.:

    DATABASE_URL=postgresql+psycopg://user:pw@localhost:5432/verdan
    DATABASE_URL=sqlite:///:memory:
"""

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base


_DEFAULT_URL = "sqlite:///./bi_forecast.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", _DEFAULT_URL)


def make_engine(url: str = None, echo: bool = False) -> Engine:
    url = url or get_database_url()
    kwargs = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # Memory DBs are per-connection by default; pin one connection so all
        # sessions see the same schema/data.
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


_engine: Engine = None
_SessionLocal: sessionmaker = None


def init_db(url: str = None, echo: bool = False) -> Engine:
    """Create engine, sessionmaker, and tables. Idempotent for the same URL."""
    global _engine, _SessionLocal
    target_url = url or get_database_url()
    current_url = str(_engine.url) if _engine is not None else None
    if _engine is None or current_url != target_url:
        _engine = make_engine(url=target_url, echo=echo)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    return _engine


def get_session_factory() -> sessionmaker:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context-managed session with commit/rollback."""
    factory = get_session_factory()
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def reset_for_tests(url: str = "sqlite:///:memory:") -> None:
    """Drop all state and reinit. Tests only."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
    init_db(url=url)
