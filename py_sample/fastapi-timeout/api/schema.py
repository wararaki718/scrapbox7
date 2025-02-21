from pydantic import BaseModel


class CustomRequest(BaseModel):
    sleeptime: int = 3


class CustomResponse(BaseModel):
    message: str
