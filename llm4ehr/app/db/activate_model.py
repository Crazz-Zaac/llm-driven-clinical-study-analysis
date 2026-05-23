from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from app.db.sqlite import get_db_session
from app.db.tables.active_models import ActiveModel


def get_active_embedding_model() -> str:
    with get_db_session() as session:
        row = session.get(ActiveModel, "embedding")
        if row is None:
            raise ValueError(
                "No embedding model selected. Pull and activate one first."
            )
        return row.model_name


def set_active_embedding_model(model_name: str, embedding_dimension: int):
    with get_db_session() as session:
        stmt = (
            insert(ActiveModel)
            .values(
                model_type="embedding",
                model_name=model_name,
                embedding_dimension=embedding_dimension,
            )
            .on_conflict_do_update(
                index_elements=["model_type"],
                set_={
                    "model_name": model_name,
                    "embedding_dimension": embedding_dimension,
                },
            )
        )
        session.execute(stmt)
        session.commit()


def get_active_embedding_dimension() -> int:
    with get_db_session() as session:
        row = session.get(ActiveModel, "embedding")
        if row is None:
            raise ValueError(
                "No embedding model selected. Pull and activate one first."
            )
        if row.embedding_dimension is None:
            raise ValueError(f"No embedding dimension stored for '{row.model_name}'.")
        return row.embedding_dimension


def get_active_chat_model() -> str:
    with get_db_session() as session:
        row = session.get(ActiveModel, "chat")
        if row is None:
            raise ValueError("No chat model selected. Pull and activate one first.")
        return row.model_name


def set_active_chat_model(model_name: str):
    with get_db_session() as session:
        stmt = (
            insert(ActiveModel)
            .values(
                model_type="chat",
                model_name=model_name,
            )
            .on_conflict_do_update(
                index_elements=["model_type"],
                set_={"model_name": model_name},
            )
        )
        session.execute(stmt)
        session.commit()
