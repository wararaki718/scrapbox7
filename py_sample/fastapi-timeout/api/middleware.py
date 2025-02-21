import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, timeout: int = 10):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return JSONResponse({"detail": "Request timeout"}, status_code=504)
