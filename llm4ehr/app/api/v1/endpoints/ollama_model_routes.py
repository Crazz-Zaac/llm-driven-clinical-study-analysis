import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.ollama_models_schema import (
    PullModelRequest,
    PullModelResponse,
    ActivateModelRequest,
    ActivateModelResponse,
)
from app.rag.services.ollama_service import OllamaService
from app.db.activate_model import (
    set_active_embedding_model,
    set_active_chat_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model Management"])


@router.post("/pull", response_model=PullModelResponse, tags=["Models"])
async def pull_model(request: PullModelRequest):

    try:
        OllamaService.pull_model(request.model_name)

        return {
            "success": True,
            "model_name": request.model_name,
            "model_type": request.model_type,
            "message": f"Model '{request.model_name}' downloaded successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activate", response_model=ActivateModelResponse, tags=["Models"])
async def activate_model(request: ActivateModelRequest):
    try:
        available = OllamaService.list_models()
        names = [m["name"] for m in available.get("models", [])]
        if request.model_name not in names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' is not downloaded. Pull it first.",
            )

        if request.model_type == "embedding":
            dimension = OllamaService.get_embedding_dimension(request.model_name)
            set_active_embedding_model(request.model_name, dimension)
        elif request.model_type == "chat":
            set_active_chat_model(request.model_name)
        else:
            raise HTTPException(
                status_code=400, detail="model_type must be 'embedding' or 'chat'"
            )

        return ActivateModelResponse(
            success=True,
            model_name=request.model_name,
            model_type=request.model_type,
            message=f"'{request.model_name}' is now the active {request.model_type} model",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", tags=["Models"])
async def list_models():

    try:
        return OllamaService.list_models()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete", tags=["Models"])
async def delete_model(model_name: str):

    try:
        OllamaService.delete_model(model_name)
        return {
            "success": True,
            "message": f"Model '{model_name}' deleted successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
