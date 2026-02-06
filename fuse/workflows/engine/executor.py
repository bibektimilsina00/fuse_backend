import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import Session
from fuse.database import engine as db_engine
from fuse.workflows.engine.error_handler import (
    ErrorClassifier,
    ErrorContext,
    ErrorPolicyHandler,
    RetryHandler,
)
from fuse.workflows.engine.nodes.registry import NodeRegistry
from fuse.workflows.engine.state import WorkflowState
from fuse.workflows.engine.context import NodeContext
from fuse.workflows.logger import WorkflowExecutorLogger
from fuse.workflows.models import NodeExecution, Workflow, WorkflowExecution
from fuse.workflows.types import TriggerDataDict

logger = logging.getLogger(__name__)


async def run_node_logic(execution_id: uuid.UUID, node_execution_id: uuid.UUID):
    """
    Core logic for executing a single node.
    """
    state = WorkflowState(None, execution_id)
    node = None
    node_config = {}

    with Session(db_engine) as session:
        node_execution = session.get(NodeExecution, node_execution_id)
        if not node_execution:
            return

        workflow_execution = session.get(WorkflowExecution, execution_id)
        workflow_id = workflow_execution.workflow_id
        state.workflow_id = workflow_id

        exec_logger = WorkflowExecutorLogger(workflow_id, execution_id)
        input_data = (
            json.loads(node_execution.input_data) if node_execution.input_data else {}
        )
        state.update_node_status(session, node_execution_id, "running")
        exec_logger.log_node_start(
            node_execution.node_id, str(node_execution.id), input_data=input_data
        )

        workflow = session.get(Workflow, workflow_id)
        node = next(
            (n for n in workflow.nodes if n.node_id == node_execution.node_id), None
        )

        if not node:
            state.update_node_status(
                session, node_execution_id, "failed", error="Node definition not found"
            )
            exec_logger.log_node_failed(
                node_execution.node_id, "Node definition not found"
            )
            return

        from sqlmodel import select

        completed_executions = session.exec(
            select(NodeExecution)
            .where(NodeExecution.workflow_execution_id == execution_id)
            .where(NodeExecution.status == "completed")
        ).all()

        results_map = {}
        if workflow_execution.trigger_data:
            try:
                t_data = json.loads(workflow_execution.trigger_data)
                results_map["trigger"] = t_data
                if isinstance(t_data, dict):
                    results_map.update(t_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse trigger_data as JSON: {e}")

        for ex in completed_executions:
            if ex.output_data:
                try:
                    data = json.loads(ex.output_data)
                    results_map[ex.node_id] = data
                    if isinstance(data, dict):
                        results_map.update(data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse output_data for node {ex.node_id}: {e}"
                    )

        node_config = node.spec.get("config", {}) if node.spec else {}
        node_settings = node.spec.get("settings", {}) if node.spec else {}
        error_policy = node_settings.get(
            "on_error", node_config.get("error_policy", "stop")
        )
        retry_config = node_settings.get("retry", {})
        max_retries = (
            retry_config.get("max_attempts", 0) if error_policy == "retry" else 0
        )

    try:
        node_pkg = NodeRegistry.get_node(node.node_type)
        if not node_pkg:
            raise ValueError(f"Unknown node type: {node.node_type}")

        node_ctx = NodeContext(
            execution_id=str(execution_id),
            workflow_id=str(workflow_id),
            node_id=node.node_id,
            config=node_config,
            input_data=input_data,
            results_map=results_map,
            edges=workflow.edges,
            env=dict(os.environ)
        )
        
        resolved_config = node_ctx.resolve_config()
        
        context = {
            "workflow_id": str(workflow_id),
            "execution_id": str(execution_id),
            "node_id": node.node_id,
            "node_config": resolved_config,
            "results": results_map,
        }

        logger.info(f"Executing node {node.node_id} ({node.node_type})...")
        # Wrapper to adapt new execute_node to RetryHandler signature
        async def _execute_pro(ctx_: Dict, inp_: Any):
            return await NodeRegistry.execute_node(
                node_id=node.node_type,
                config=resolved_config,
                inputs=inp_,
                credentials=None,
                context=node_ctx # Pass full context
            )

        if max_retries > 0:
            result, attempts = await RetryHandler.execute_with_retry(
                _execute_pro, context, input_data, max_retries=max_retries
            )
        else:
            result = await NodeRegistry.execute_node(
                node_id=node.node_type,
                config=resolved_config, # Fixed: Use resolved config here too
                inputs=input_data,
                credentials=None,
                context=node_ctx # Pass full context
            )

        with Session(db_engine) as session:
            state.update_node_status(
                session, node_execution_id, "completed", result=result
            )
            exec_logger.log_node_complete(node.node_id, result)

        from fuse.workflows.engine.scheduler import WorkflowScheduler
        scheduler = WorkflowScheduler(workflow_id, execution_id)
        scheduler.handle_node_completion(node_execution_id, result)

    except Exception as e:
        error_context = ErrorClassifier.classify(e)
        logger.error(
            f"Error executing node {node.node_id}: {error_context.message}"
        )

        with Session(db_engine) as session:
            error_info = {
                "message": error_context.message,
                "category": error_context.category.value,
                "suggestion": error_context.suggestion,
                "details": error_context.original_error,
            }
            error_str = json.dumps(error_info)

            try:
                state.update_node_status(
                    session, node_execution_id, "failed", error=error_str
                )
            except Exception as transition_error:
                logger.warning(f"Could not update node status to failed: {transition_error}")
                
            exec_logger.log_node_failed(node.node_id, error_context.message, error_context.to_dict())

            should_continue = ErrorPolicyHandler.should_continue_workflow(
                error_context, error_policy, "stop"
            )

            if should_continue:
                fallback_output = ErrorPolicyHandler.get_fallback_output(error_context)
                from fuse.workflows.engine.scheduler import WorkflowScheduler
                scheduler = WorkflowScheduler(workflow_id, execution_id)
                scheduler.handle_node_completion(node_execution_id, fallback_output)
            else:
                try:
                    state.update_execution_status(session, "failed")
                    workflow_execution = state.get_execution(session)
                    workflow_execution.error = (
                        f"Node {node.node_id} failed: {error_context.message}"
                    )
                    session.add(workflow_execution)
                    session.commit()
                except Exception as ef:
                    logger.error(f"Double fault: Could not mark workflow as failed: {ef}")
                
                exec_logger.log_workflow_failed(error_context.message)


async def execute_workflow_task(
    workflow_id: str,
    trigger_data: Optional[TriggerDataDict] = None,
    execution_id: str = None,
):
    """Entry point to execute a workflow locally."""
    from fuse.workflows.engine.constants import ExecutionStatus
    from fuse.workflows.engine.scheduler import WorkflowScheduler

    workflow_uuid = uuid.UUID(workflow_id)
    execution_uuid = uuid.UUID(execution_id) if execution_id else None

    if not execution_uuid:
        with Session(db_engine) as session:
            execution = WorkflowExecution(
                workflow_id=workflow_uuid,
                status=ExecutionStatus.PENDING.value,
                trigger_data=json.dumps(trigger_data) if trigger_data else None,
            )
            session.add(execution)
            session.commit()
            session.refresh(execution)
            execution_uuid = execution.id

    scheduler = WorkflowScheduler(workflow_uuid, execution_uuid)
    scheduler.start_workflow(trigger_data)
    return str(execution_uuid)


class WorkflowExecutor:
    @staticmethod
    def dispatch_node_task(
        execution_id: uuid.UUID, node_execution_id: uuid.UUID, node_type: str
    ):
        """Runs the node logic in-process as an async task."""
        logger.info(f"Local Execution: Running node {node_type} ({node_execution_id})")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(run_node_logic(execution_id, node_execution_id))
            else:
                loop.run_until_complete(run_node_logic(execution_id, node_execution_id))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_node_logic(execution_id, node_execution_id))
