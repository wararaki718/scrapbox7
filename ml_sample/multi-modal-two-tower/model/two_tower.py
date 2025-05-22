import torch

from config import DocumentTowerConfig, QueryTowerConfig
from .document_tower import DocumentTower
from .query_tower import QueryTower


class TwoTowerModel(torch.nn.Module):
    def __init__(self, query_tower_config: QueryTowerConfig, document_tower_config: DocumentTowerConfig) -> None:
        super().__init__()
        self.query_tower = QueryTower(query_tower_config)
        self.document_tower = DocumentTower(document_tower_config)

    def forward(
        self,
        # query
        x_query_context: torch.Tensor,
        x_query_image: torch.Tensor,
        x_query_text: torch.Tensor,
        x_query_action: torch.Tensor,
        # document
        x_document_context: torch.Tensor,
        x_document_image: torch.Tensor,
        x_document_text: torch.Tensor,
        x_document_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_output = self.query_tower(
            x_query_context, x_query_image, x_query_text, x_query_action
        )
        document_output = self.document_tower(
            x_document_context, x_document_image, x_document_text, x_document_action
        )
        return query_output, document_output
