from typing import Any, Dict, List
from fuse.workflows.engine.context import NodeContext
from fuse.workflows.engine.definitions import WorkflowItem

async def execute(context: NodeContext) -> List[WorkflowItem]:
    """Provide configuration for the model."""
    config = context.resolve_config()
    
    cred_id = config.get("credential")
    model_name = config.get("model")
    
    if not cred_id or not model_name:
        raise ValueError("Missing configuration (credential or model)")
        
    return [WorkflowItem(
        json={
            "model": {
                "credential_id": cred_id,
                "model_name": model_name,
                "temperature": config.get("temperature", 0.7)
            }
        }
    )]

async def validate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration"""
    errors = []
    if not config.get("credential"): errors.append("Credential is required")
    if not config.get("model"): errors.append("Model is required")
    return {"valid": len(errors) == 0, "errors": errors}
