import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import uuid

from fuse.workflows.engine.nodes.registry import NodeRegistry
from .store import VectorStore
from .store import VectorStore
from .chroma_store import ChromaVectorStore

# Try to use Chroma if configured, else fallback
# For now, we prefer SimpleStore until deps are confirmed, but allow config override.
USE_CHROMA = True

logger = logging.getLogger(__name__)

class NodeRAGService:
    """
    Orchestrates the indexing and retrieval of nodes for AI context.
    Acts as the 'brain' that decides what tools the AI sees.
    """
    
    _instance = None
    
    def __init__(self):
        self.store: VectorStore = ChromaVectorStore()
        self.last_indexed: Optional[datetime] = None
        self._index_lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NodeRAGService()
        return cls._instance
        
    async def ensure_index(self, force: bool = False):
        """
        Check if index needs rebuilding.
        Rebuilds if empty or force=True.
        """
        async with self._index_lock:
            # Simple check: if store is empty or force
            # SimpleStore is in-memory, so it's always empty on restart.
            # Chroma persists, so we might check if collection exists.
            
            # Since SimpleStore is primary for now, let's always index on startup.
            if not self.last_indexed or force:
                logger.info("Building Node RAG Index...")
                # print("DEBUG: Building Node RAG Index...", flush=True)
                
                # Fetch all nodes from Registry (Official)
                # TODO: Retrieve Custom Nodes from DB too
                nodes = NodeRegistry.get_all_schemas()
                # print(f"DEBUG: RAG Indexing {len(nodes)} nodes...", flush=True)
                if nodes:
                    # print(f"DEBUG: First Node Keys: {list(nodes[0].keys())}", flush=True)
                    # print(f"DEBUG: First Node ID: {nodes[0].get('id')}", flush=True)
                    pass

                # Clear old index
                await self.store.clear()
                
                # Add to index
                await self.store.add_nodes(nodes)
                
                self.last_indexed = datetime.now()
                logger.info(f"Indexed {len(nodes)} nodes.")
                
    async def retrieve_relevant_nodes(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant nodes.
        Returns full schema dicts.
        """
        await self.ensure_index()
        
        # Search index
        # Results might be lightweight metadata or full objects depending on store impl
        results = await self.store.search(query, limit=limit)
        # print(f"DEBUG: RAG Store Search Raw Results Count: {len(results)}", flush=True)
        
        # If store returns only metadata/IDs, fetch full schema from Registry
        full_schemas = []
        all_nodes = NodeRegistry.list_nodes()
        # print(f"DEBUG: Registry Keys: {list(all_nodes.keys())[:5]}...", flush=True)

        for res in results:
            node_id = res.get("id")
            # print(f"DEBUG: Processing Result ID: {node_id}", flush=True)
            if node_id:
                # We fetch fresh from registry to ensure latest config
                # TODO: Optimize?
                try:
                    if node_id in all_nodes:
                        full_schemas.append(all_nodes[node_id])
                    else:
                        # Maybe it was a custom node not in registry cache?
                        # Fallback to result if it looks complete
                        # print(f"DEBUG: Node ID {node_id} NOT found in registry keys.", flush=True)
                        if "inputs" in res:
                            full_schemas.append(res)
                except Exception as e:
                    logger.warning(f"Failed to fetch detailed schema for {node_id}: {e}")
                    
        return full_schemas
