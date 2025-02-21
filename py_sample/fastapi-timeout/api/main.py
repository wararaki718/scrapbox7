from time import sleep

from fastapi import FastAPI

from .middleware import TimeoutMiddleware
from .schema import CustomRequest, CustomResponse

app = FastAPI()
app.add_middleware(TimeoutMiddleware, timeout=5)


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/sample", response_model=CustomResponse)
def sample(request: CustomRequest) -> CustomResponse:
    sleep(request.sleeptime)
    return CustomResponse(message="ok")

