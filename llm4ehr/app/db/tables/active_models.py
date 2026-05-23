from datetime import datetime, timezone, timezone
from sqlalchemy import Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.sqlite import Base


class ActiveModel(Base):
    __tablename__ = "active_models"

    model_type: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimension: Mapped[int | None] = mapped_column(
        nullable=True
    )  # None for chat models
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    activated: Mapped[bool] = mapped_column(default=True, nullable=False)
