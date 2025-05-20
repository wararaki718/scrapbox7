from pydantic import BaseModel

class QueryTowerConfig(BaseModel):
    text_input_dim: int = 128
    image_input_dim: int = 64
    context_input_dim: int = 32
    action_input_dim: int = 64

    text_output_dim: int = 16
    image_output_dim: int = 16
    context_output_dim: int = 16
    action_output_dim: int = 16

    output_dim: int = 32
    dropout: float = 0.1
