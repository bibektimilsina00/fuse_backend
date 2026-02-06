import functools
import logging
import os
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger("fuse_backend")

# Bypass caching completely
DISABLE_CACHE = True

# Type variable for generic return types
T = TypeVar("T")

class CacheTTL:
    """Standard cache TTL values."""
    SHORT = 60
    MEDIUM = 300
    LONG = 3600
    NODE_TYPES = 60
    WORKFLOW_META = 300
    USER_SESSION = 86400

def cache(
    ttl: int = CacheTTL.MEDIUM,
    prefix: Optional[str] = None,
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """No-op cache decorator (Redis removed)."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        return wrapper
    return decorator

F = TypeVar("F", bound=Callable[..., Any])

def async_cache(
    ttl: int = CacheTTL.MEDIUM,
    prefix: Optional[str] = None,
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[F], F]:
    """No-op async cache decorator (Redis removed)."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator

def invalidate_cache(pattern: str) -> int:
    return 0

async def async_invalidate_cache(pattern: str) -> int:
    return 0

def invalidate_node_types_cache() -> int:
    return 0

def invalidate_workflow_cache(workflow_id: str) -> int:
    return 0

def invalidate_user_cache(user_id: str) -> int:
    return 0
