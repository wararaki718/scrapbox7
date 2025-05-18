from pydantic import BaseModel


class DocumentTowerConfig(BaseModel):
    text_input_dim: int = 128
    image_input_dim: int = 64
    context_input_dim: int = 32
    actions_input_dim: int = 64

    text_output_dim: int = 16
    image_output_dim: int = 16
    context_output_dim: int = 16
    actions_output_dim: int = 16

    output_dim: int = 32
    dropout: float = 0.1
