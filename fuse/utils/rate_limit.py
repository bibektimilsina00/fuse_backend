import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    No-op rate limiting middleware (Redis removed).
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        return await call_next(request)
