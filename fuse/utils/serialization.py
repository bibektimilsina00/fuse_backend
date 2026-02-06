from typing import Any
import json
from pydantic import BaseModel

def to_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serializable format.
    Handles Pydantic models (V1 and V2).
    """
    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
        if hasattr(obj, "model_dump"):
            return to_serializable(obj.model_dump(by_alias=True))
        return to_serializable(obj.dict(by_alias=True))
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    return obj

class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            if hasattr(obj, "model_dump"):
                return obj.model_dump(by_alias=True)
            return obj.dict(by_alias=True)
        return super().default(obj)
