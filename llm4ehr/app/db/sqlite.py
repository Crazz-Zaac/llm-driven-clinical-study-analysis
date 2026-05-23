from contextlib import contextmanager
from sqlalchemy import create_engine
from collections.abc import Generator
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.SQLITE_DB_PATH,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Use this anywhere outside FastAPI route functions."""
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# FastAPI dependency — for use in route functions only
def get_db():
    with get_db_session() as session:
        yield session


def init_db():
    """Creates all tables if they don't exist. Called once at app startup."""
    from app.db.tables import (
        document,
        active_models,
    )  # noqa: F401 — registers models with Base

    Base.metadata.create_all(bind=engine)
