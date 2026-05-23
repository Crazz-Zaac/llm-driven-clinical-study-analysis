from pydantic import BaseModel
from typing import Literal


class PullModelRequest(BaseModel):
    model_name: str
    model_type: Literal["embedding", "chat"]


class ActivateModelRequest(BaseModel):
    model_name: str
    model_type: Literal["embedding", "chat"]

class ActivateModelResponse(BaseModel):
    success: bool
    model_name: str
    model_type: str
    message: str


class PullModelResponse(BaseModel):
    success: bool
    model_name: str
    model_type: str
    message: str
