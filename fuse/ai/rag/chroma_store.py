import logging
import asyncio
from typing import Any, Dict, List, Optional
import os
import shutil

from .store import VectorStore

logger = logging.getLogger(__name__)

# Constants
CHROMA_PERSIST_DIR = "fuse_vector_index"
COLLECTION_NAME = "fuse_nodes"

class ChromaVectorStore(VectorStore):
    """
    A persistent, scalable vector store using ChromaDB.
    Requires: chromadb, sentence-transformers
    """
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError("chromadb is not installed. Please install it with `pip install chromadb sentence-transformers`.")
            
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self.chroma_lib = chromadb
        self.embedding_fn = None
        self.init_lock = asyncio.Lock()
        self.initialized = False
        
    async def initialize(self):
        if self.initialized:
            return
            
        async with self.init_lock:
            # We use synchronous library calls inside async wrapper
            try:
                # Initialize Chroma Client (Persistent)
                self.client = self.chroma_lib.PersistentClient(path=self.persist_dir)
                
                # Use default sentence-transformer embedding (all-MiniLM-L6-v2)
                # This downloads model once (~80MB)
                self.embedding_fn = self.chroma_lib.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                
                logger.info(f"ChromaVectorStore initialized at {self.persist_dir}")
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise e

    async def clear(self):
        if not self.client:
            await self.initialize()
        
        # Delete and recreate collection
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB: {e}")

    async def add_nodes(self, nodes: List[Dict[str, Any]]):
        if not self.collection:
            await self.initialize()
            
        if not nodes:
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for node in nodes:
            # Use 'id' if present, else 'name' (often used as ID in V2), else index
            node_id = node.get("id") or node.get("name") or str(len(ids))
            name = node.get("name", "")
            desc = node.get("description", "")
            tags = ", ".join(node.get("tags", []))
            kind = node.get("type", "unknown")
            
            # Text to embed
            doc_text = f"Node: {name}\nID: {node_id}\nType: {kind}\nDescription: {desc}\nTags: {tags}"
            
            ids.append(node_id)
            documents.append(doc_text)
            
            # Store full node JSON as metadata string (or selected fields)
            # Chroma metadata must be simple types (str, int, float, bool)
            meta = {
                "id": node_id,
                "name": name,
                "kind": kind,
                # "full_json": json.dumps(node) # Store full json if needed
                "description": desc[:1000] # Truncate metadata if too long
            }
            metadatas.append(meta)
            
        # Add to collection (batch logic handled by Chroma usually, but let's batch manually if huge)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
            
        logger.info(f"Added {len(nodes)} nodes to ChromaDB index.")

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.collection:
            await self.initialize()
            
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Results structure: {'ids': [['id1', ...]], 'metadatas': [[{...}, ...]], ...}
        # Be careful with nested lists (one per query)
        
        found_nodes = []
        if results['ids']:
            ids = results['ids'][0]
            metas = results['metadatas'][0]
            
            for i, node_id in enumerate(ids):
                # Typically we'd fetch the full node from Registry because metadata is limited
                # But we return what we have in metadata + id for the retriever to look up
                meta = metas[i]
                found_nodes.append(meta)
                
        return found_nodes
