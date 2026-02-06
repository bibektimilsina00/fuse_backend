import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class WorkflowExecutorLogger:
    """
    Handles logging of workflow execution events to Memory PubSub (for real-time frontend updates).
    """
    def __init__(self, workflow_id: uuid.UUID, execution_id: uuid.UUID):
        from fuse.utils.memory_pubsub import memory_pubsub
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.memory_pubsub = memory_pubsub
        self.channel = f"workflow:execution:{execution_id}"

    def _publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event to Memory Pub/Sub."""
        message = {
            "type": event_type,
            "timestamp": str(datetime.utcnow()),
            "data": data
        }
        
        # Fire and forget mechanism for sync contexts
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            
            if loop and loop.is_running():
                asyncio.create_task(self.memory_pubsub.publish(self.channel, message))
            else:
                # If we are in a sync thread (very rare now with local exec), 
                # we rely on MemoryPubSub's run_coroutine_threadsafe handling
                 asyncio.run(self.memory_pubsub.publish(self.channel, message))
        except Exception as e:
             # Don't let logging errors crash execution
             logger.error(f"Logger error: {e}")

    def log_workflow_start(self):
        self._publish("workflow_started", {
            "workflow_id": str(self.workflow_id),
            "execution_id": str(self.execution_id)
        })

    def log_workflow_complete(self):
        self._publish("workflow_completed", {
            "workflow_id": str(self.workflow_id),
            "execution_id": str(self.execution_id),
            "status": "completed"
        })

    def log_workflow_failed(self, error: str):
        self._publish("workflow_failed", {
            "workflow_id": str(self.workflow_id),
            "execution_id": str(self.execution_id),
            "error": error
        })

    def log_node_scheduled(self, node_id: str, node_execution_id: str):
        self._publish("node_scheduled", {
            "node_id": node_id,
            "node_execution_id": node_execution_id
        })

    def log_node_start(self, node_id: str, node_execution_id: str, input_data: Any = None):
        self._publish("node_started", {
            "node_id": node_id,
            "node_execution_id": node_execution_id,
            "input_data": input_data
        })

    def log_node_complete(self, node_id: str, result: Any):
        self._publish("node_completed", {
            "node_id": node_id,
            "result": result
        })

    def log_node_failed(self, node_id: str, error: str, error_context: dict = None):
        data = {
            "node_id": node_id,
            "error": error
        }
        if error_context:
            data["error_category"] = error_context.get("category", "unknown")
            data["error_suggestion"] = error_context.get("suggestion")
            data["is_retryable"] = error_context.get("is_retryable", False)
        self._publish("node_failed", data)
    
    def log_node_retrying(self, node_id: str, attempt: int, max_attempts: int, delay: float):
        self._publish("node_retrying", {
            "node_id": node_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "delay_seconds": delay
        })
    
    def log_node_continued(self, node_id: str, error: str):
        self._publish("node_continued", {
            "node_id": node_id,
            "error": error,
            "message": "Node failed but workflow continues (error_policy='continue')"
        })
