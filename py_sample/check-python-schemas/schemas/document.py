from pydantic import BaseModel


class Document(BaseModel):
    doc_id: int
    title: str
    price: int
