from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

engine = create_engine(
    settings.SQLITE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite only
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist. Called once at app startup."""
    from app.db.tables import document  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)