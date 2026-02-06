import asyncio
import logging
from typing import Dict, Set, Any
import json

logger = logging.getLogger(__name__)

class MemoryPubSub:
    """
    An in-memory Pub/Sub replacement for Redis.
    Used for local execution mode where Redis is not available.
    Ensures thread-safety by scheduling operations on the main event loop.
    """
    def __init__(self):
        self.subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self.history: Dict[str, List[str]] = {}
        self.main_loop = None
        
        # Capture the main loop when instantiated if possible
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None

    def ensure_loop(self):
        """Ensure we have a reference to the main loop."""
        if not self.main_loop:
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        return self.main_loop

    async def subscribe(self, channel: str) -> asyncio.Queue:
        """Subscribe to a channel and return a queue for receiving messages."""
        self.ensure_loop()
        queue = asyncio.Queue()
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        self.subscribers[channel].add(queue)
        
        # Replay history
        if channel in self.history:
            for msg in self.history[channel]:
                await queue.put(msg)
                
        logger.debug(f"Subscribed to channel {channel}. Total subscribers: {len(self.subscribers[channel])}")
        return queue

    async def unsubscribe(self, channel: str, queue: asyncio.Queue):
        """Unsubscribe from a channel."""
        if channel in self.subscribers:
            self.subscribers[channel].discard(queue)
            if not self.subscribers[channel]:
                del self.subscribers[channel]
                # Optional: Clear history if no one is listening anymore? 
                # Better to keep it for a bit in case of page refresh.
                # For now, we rely on restart to clear memory.
            logger.debug(f"Unsubscribed from channel {channel}")

    async def _publish_internal(self, channel: str, message_str: str):
        """Internal publish logic running on the main loop."""
        # Store in history
        if channel not in self.history:
            self.history[channel] = []
        self.history[channel].append(message_str)
        # Limit history size per channel execution is likely short lived in dev
        if len(self.history[channel]) > 100:
            self.history[channel].pop(0)

        if channel in self.subscribers:
            # logger.debug(f"Publishing to {channel} for {len(self.subscribers[channel])} subscribers")
            for q in list(self.subscribers[channel]): # Copy set to avoid size change iteration
                await q.put(message_str)

    async def publish(self, channel: str, message: Any):
        """Publish a message to all subscribers of a channel."""
        if isinstance(message, (dict, list)):
            message_str = json.dumps(message)
        else:
            message_str = str(message)

        main_loop = self.ensure_loop()
        
        # Check if we are running in the main loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if main_loop and current_loop != main_loop:
            # We are in a different thread/loop, verify main loop is running
            if main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._publish_internal(channel, message_str), 
                    main_loop
                )
                # We generally don't wait for result here to avoid blocking
                return
        
        # We are on the main loop (or main loop is not captured), just await directly
        await self._publish_internal(channel, message_str)

memory_pubsub = MemoryPubSub()
