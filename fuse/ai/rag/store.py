from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

class VectorStore(ABC):
    """
    Abstract Base Class for Vector Store implementations.
    Allows swapping between simple keyword search and heavy Chroma/FAISS search.
    """

    @abstractmethod
    async def initialize(self):
        """Initialize connection, tables, or collections."""
        pass

    @abstractmethod
    async def add_nodes(self, nodes: List[Dict[str, Any]]):
        """Add nodes to the index."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant nodes."""
        pass

    @abstractmethod
    async def clear(self):
        """Clear the index."""
        pass
