import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from fuse.auth.dependencies import CurrentUser, SessionDep
from fuse.utils.cache import CacheTTL, cache
from fuse.workflows.code_execution import router as code_execution_router
from fuse.workflows.models import WorkflowExecution
from fuse.workflows.schemas import (
    AIWorkflowRequest,
    AIWorkflowResponse,
    ExecuteNodeRequest,
    ExecuteNodeResponse,
    Message,
    TriggerWebhookResponse,
    WorkflowCreate,
    WorkflowExecutionPublic,
    WorkflowPublic,
    WorkflowSaveRequest,
    WorkflowsPublic,
    WorkflowUpdate,
)
from fuse.workflows.service import workflow_service
from starlette.concurrency import run_in_threadpool

router = APIRouter()

# Include code execution sub-router
router.include_router(code_execution_router, tags=["code-execution"])


@router.post("/{id}/execute", response_model=WorkflowExecutionPublic)
def execute_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
    trigger_data: Dict[str, Any] = {},
    background_tasks: BackgroundTasks = None,
) -> Any:
    try:
        workflow_uuid = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")
    
    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_uuid)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Create execution record first so we can return the ID immediately
    execution = WorkflowExecution(
        workflow_id=workflow_uuid, status="pending", trigger_data=str(trigger_data)
    )
    session.add(execution)
    session.commit()
    session.refresh(execution)

    # Local In-Process Execution
    from fuse.workflows.engine import execute_workflow as execute_workflow_task
    
    # Mark as manual execution to allow engine to skip triggers if needed
    trigger_data = dict(trigger_data)
    trigger_data["__manual"] = True
    
    async def safe_execute():
        try:
            await execute_workflow_task(str(workflow_uuid), trigger_data, str(execution.id))
        except Exception as e:
            logger.exception(f"Fatal error during workflow execution start: {e}")
            # Try to mark the whole execution as failed if it hasn't started any nodes
            with Session(db_engine) as session_fail:
                exec_rec = session_fail.get(WorkflowExecution, execution.id)
                if exec_rec and exec_rec.status not in ["completed", "failed"]:
                    exec_rec.status = "failed"
                    exec_rec.error = f"Initialization error: {str(e)}"
                    session_fail.add(exec_rec)
                    session_fail.commit()

    logger.info(f"Triggering Local Execution for workflow {workflow_uuid}")
    if background_tasks:
        background_tasks.add_task(safe_execute)
    else:
        # Fallback to direct task creation
        asyncio.create_task(safe_execute())

    return execution


@router.websocket("/ws/{execution_id}")
async def websocket_endpoint(websocket: WebSocket, execution_id: str):
    await websocket.accept()
    logger.debug(f"WebSocket connected (Execution: {execution_id}, Mode: Memory)")
    
    # Send an initial message to confirm connection
    await websocket.send_json({
        "type": "info", 
        "timestamp": str(datetime.utcnow()),
        "data": {"message": "Connected to local log stream"}
    })

    # Memory PubSub Mode
    from fuse.utils.memory_pubsub import memory_pubsub
    channel = f"workflow:execution:{execution_id}"
    queue = await memory_pubsub.subscribe(channel)
    logger.debug(f"Subscribed to Memory channel: {channel}")
    
    try:
        while True:
            message = await queue.get()
            await websocket.send_text(message)
    except WebSocketDisconnect:
        pass
    finally:
        await memory_pubsub.unsubscribe(channel, queue)


@router.get("/executions/{id}", response_model=WorkflowExecutionPublic)
def get_execution(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """
    Get workflow execution by ID.
    """
    execution = session.get(WorkflowExecution, id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if execution.workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return execution


@router.get("/", response_model=WorkflowsPublic)
def read_workflows(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve workflows owned by current user.
    """
    workflows = workflow_service.get_workflows_by_owner(
        session=session, owner_id=current_user.id, skip=skip, limit=limit
    )
    count = workflow_service.count_workflows_by_owner(
        session=session, owner_id=current_user.id
    )

    return WorkflowsPublic(
        data=[workflow_service.workflow_to_public(w) for w in workflows], count=count
    )


@router.get("/new")
def read_new_workflow():
    """Handle 'new' placeholder to avoid UUID validation errors."""
    raise HTTPException(status_code=404, detail="Workflow template not found")


@router.get("/{id}", response_model=WorkflowPublic)
def read_workflow(
    id: str,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """
    Get workflow by ID.
    """
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return workflow_service.workflow_to_public(workflow)


@router.post("/", response_model=WorkflowPublic)
def create_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    workflow_in: WorkflowCreate,
) -> Any:
    """
    Create new workflow.
    """
    workflow = workflow_service.create_workflow(
        session=session, workflow_in=workflow_in, owner_id=current_user.id
    )
    return workflow_service.workflow_to_public(workflow)


@router.patch("/{id}", response_model=WorkflowPublic)
def update_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
    workflow_in: WorkflowUpdate,
) -> Any:
    """
    Update a workflow.
    """
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    workflow = workflow_service.update_workflow(
        session=session, db_workflow=workflow, workflow_in=workflow_in
    )
    return workflow_service.workflow_to_public(workflow)


@router.delete("/{id}", response_model=Message)
def delete_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
) -> Any:
    """
    Delete a workflow.
    """
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    workflow_service.delete_workflow(session=session, workflow_id=workflow_id)
    return Message(message="Workflow deleted successfully")


@router.post("/{id}/activate", response_model=WorkflowPublic)
def activate_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
) -> Any:
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    workflow.status = "active"
    workflow.updated_at = datetime.utcnow()
    session.add(workflow)
    session.commit()
    session.refresh(workflow)

    return workflow_service.workflow_to_public(workflow)


@router.post("/{id}/deactivate", response_model=WorkflowPublic)
def deactivate_workflow(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
) -> Any:
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    workflow.status = "inactive"
    workflow.updated_at = datetime.utcnow()
    session.add(workflow)
    session.commit()
    session.refresh(workflow)

    return workflow_service.workflow_to_public(workflow)


@router.post("/{id}/save", response_model=WorkflowPublic)
def save_workflow_nodes(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
    save_request: WorkflowSaveRequest,
) -> Any:
    """
    Save workflow nodes and edges.
    """
    if id == "new":
        workflow_in = WorkflowCreate(
            name=save_request.meta.name or "Untitled Workflow",
            description=save_request.meta.description,
            status=save_request.meta.status or "draft",
        )
        workflow = workflow_service.create_workflow(
            session=session, workflow_in=workflow_in, owner_id=current_user.id
        )
        workflow_id = workflow.id
    else:
        try:
            workflow_id = uuid.UUID(id)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = workflow_service.get_workflow(session=session, workflow_id=workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    workflow = workflow_service.save_workflow_nodes(
        session=session, workflow_id=workflow_id, save_request=save_request
    )
    return workflow_service.workflow_to_public(workflow)


@cache(ttl=CacheTTL.NODE_TYPES, prefix="node_types")
def _get_node_types_cached() -> List[Dict[str, Any]]:
    """Cached helper to get node type schemas."""
    import fuse.workflows.engine.nodes
    from fuse.workflows.engine.nodes.registry import NodeRegistry

    schemas = NodeRegistry.get_all_schemas()
    return [s if isinstance(s, dict) else s.model_dump() for s in schemas]


@router.get("/nodes/types", response_model=List[Dict[str, Any]])
def get_node_types(
    current_user: CurrentUser,
) -> Any:
    """
    Get all available node types and their schemas.
    """
    return _get_node_types_cached()


class NodeOptionsRequest(BaseModel):
    node_type: str
    method_name: str
    dependency_values: Dict[str, Any]


@router.post("/node/options", response_model=List[Dict[str, str]])
async def get_node_options(
    request: NodeOptionsRequest,
    current_user: CurrentUser,
) -> Any:
    """
    Fetch dynamic options for a node input.
    """
    logger.debug(f"get_node_options called with: {request}")
    from fuse.workflows.engine.nodes.registry import NodeRegistry

    node_pkg = NodeRegistry.get_node(request.node_type)
    if not node_pkg:
        raise HTTPException(
            status_code=400, detail=f"Unknown node type: {request.node_type}"
        )

    method_name = request.method_name
    execute_fn = node_pkg.execute_fn
    
    if not execute_fn or not hasattr(execute_fn, "__globals__"):
         raise HTTPException(
            status_code=500, detail="Cannot inspect node module"
        )
        
    method = execute_fn.__globals__.get(method_name)

    if not method:
        raise HTTPException(
            status_code=400,
            detail=f"Method {method_name} not found on node {request.node_type}",
        )

    if not callable(method):
        raise HTTPException(
            status_code=400, detail=f"Attribute {method_name} is not callable"
        )

    context = {"user_id": str(current_user.id)}

    try:
        options = await method(context, request.dependency_values)
        return options
    except Exception as e:
        logger.exception(f"Error fetching options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/{workflow_id}", response_model=TriggerWebhookResponse)
async def trigger_webhook(
    workflow_id: uuid.UUID,
    request: Request,
    session: SessionDep,
) -> TriggerWebhookResponse:
    """
    Trigger a workflow via webhook.
    """
    workflow = await run_in_threadpool(
        workflow_service.get_workflow_with_nodes,
        session=session,
        workflow_id=workflow_id,
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Workflow is not active. Activate it to enable webhook triggers.",
        )

    import json
    nodes = workflow.nodes
    webhook_node = None
    for node in nodes:
        if node.node_type == "webhook.receive":
            webhook_node = node
            break

    if not webhook_node:
        raise HTTPException(
            status_code=400, detail="Workflow does not have a webhook trigger"
        )

    body = (
        await request.json()
        if request.headers.get("content-type") == "application/json"
        else await request.body()
    )
    if isinstance(body, bytes):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = body.decode("utf-8", errors="replace")

    headers = dict(request.headers)
    query = dict(request.query_params)
    method = request.method

    trigger_data = {
        "body": body,
        "headers": headers,
        "query": query,
        "method": method,
        "timestamp": str(datetime.utcnow()),
    }

    from fuse.workflows.engine import execute_workflow as execute_workflow_task

    execution = WorkflowExecution(
        workflow_id=workflow_id, status="pending", trigger_data=json.dumps(trigger_data)
    )
    session.add(execution)
    session.commit()
    session.refresh(execution)

    # Local Background Task
    asyncio.create_task(execute_workflow_task(str(workflow_id), trigger_data, str(execution.id)))

    return TriggerWebhookResponse(
        execution_id=execution.id,
        status="pending",
        message="Webhook received and workflow triggered locally",
    )


@router.post("/{id}/nodes/{node_id}/execute", response_model=ExecuteNodeResponse)
async def execute_node(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: str,
    node_id: str,
    request: ExecuteNodeRequest,
) -> ExecuteNodeResponse:
    """
    Execute a single node for testing purposes.
    """
    try:
        workflow_id = uuid.UUID(id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid workflow ID: {id}")

    workflow = await run_in_threadpool(
        workflow_service.get_workflow_with_nodes, session=session, workflow_id=workflow_id
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    node = next((n for n in workflow.nodes if n.node_id == node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    from fuse.workflows.engine.nodes.registry import NodeRegistry

    node_pkg = NodeRegistry.get_node(node.node_type)
    if not node_pkg:
        raise HTTPException(
            status_code=400, detail=f"Unknown node type: {node.node_type}"
        )

    input_data = request.input_data
    config_override = request.config

    try:
        node_config = (
            config_override
            if config_override is not None
            else (
                node.spec.get("config", {})
                if (node.spec and isinstance(node.spec, dict))
                else {}
            )
        )

        result = await NodeRegistry.execute_node(
            node_id=node.node_type,
            config=node_config,
            inputs=input_data,
            credentials=None
        )
        
        return ExecuteNodeResponse(
            status="completed",
            result=result if isinstance(result, dict) else {"_output": result},
            node_id=node_id,
        )
    except Exception as e:
        logger.exception(f"Node execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Node execution failed: {str(e)}")


@router.get("/debug/workflows")
def list_debug_workflows():
    import os
    dummy_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dummy_json"
    )
    if not os.path.exists(dummy_dir):
        return []
    files = [f for f in os.listdir(dummy_dir) if f.endswith(".json")]
    return files


@router.get("/debug/workflows/{filename}")
def get_debug_workflow(filename: str):
    import json
    import os
    dummy_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dummy_json"
    )
    file_path = os.path.join(dummy_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    with open(file_path, "r") as f:
        return json.load(f)
