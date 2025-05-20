import torch

from config import DocumentTowerConfig, QueryTowerConfig


class QueryTower(torch.nn.Module):
    def __init__(self, config: QueryTowerConfig) -> None:
        super().__init__()
        self._context_modal = torch.nn.Linear(config.context_input_dim, config.context_output_dim)
        self._text_modal = torch.nn.Linear(config.text_input_dim, config.text_output_dim)
        self._image_modal = torch.nn.Linear(config.image_input_dim, config.image_output_dim)
        self._actions_modal = torch.nn.Linear(config.action_input_dim, config.action_output_dim)

        # encoder
        self._model = torch.nn.Sequential(*[
            torch.nn.Linear(
                config.context_output_dim + config.text_output_dim + config.image_output_dim + config.action_output_dim,
                config.output_dim,
            ),
            torch.nn.Dropout(p=config.dropout),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x_context: torch.Tensor, x_image: torch.Tensor, x_text: torch.Tensor, x_action: torch.Tensor) -> torch.Tensor:
        x_context = self._context_modal(x_context)
        x_text = self._text_modal(x_text)
        x_image = self._image_modal(x_image)
        x_action = self._actions_modal(x_action)
        x = torch.cat([x_text, x_image, x_context, x_action], dim=1)
        return self._model(x)


class DocumentTower(torch.nn.Module):
    def __init__(self, config: DocumentTowerConfig) -> None:
        super().__init__()
        self._context_modal = torch.nn.Linear(config.context_input_dim, config.context_output_dim)
        self._text_modal = torch.nn.Linear(config.text_input_dim, config.text_output_dim)
        self._image_modal = torch.nn.Linear(config.image_input_dim, config.image_output_dim)
        self._actions_modal = torch.nn.Linear(config.action_input_dim, config.action_output_dim)

        # encoder
        self._model = torch.nn.Sequential(*[
            torch.nn.Linear(
                config.context_output_dim + config.text_output_dim + config.image_output_dim + config.action_output_dim,
                config.output_dim,
            ),
            torch.nn.Dropout(p=config.dropout),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x_context: torch.Tensor, x_image: torch.Tensor, x_text: torch.Tensor, x_action: torch.Tensor) -> torch.Tensor:
        x_context = self._context_modal(x_context)
        x_text = self._text_modal(x_text)
        x_image = self._image_modal(x_image)
        x_action = self._actions_modal(x_action)
        x = torch.cat([x_text, x_image, x_context, x_action], dim=1)
        return self._model(x)


class TwoTowerModel(torch.nn.Module):
    def __init__(self, query_tower_config: QueryTowerConfig, document_tower_config: DocumentTowerConfig) -> None:
        super().__init__()
        self.query_tower = QueryTower(query_tower_config)
        self.document_tower = DocumentTower(document_tower_config)

    def forward(
        self,
        x_query_context: torch.Tensor, x_query_image: torch.Tensor, x_query_text: torch.Tensor, x_query_action: torch.Tensor,
        x_document_context: torch.Tensor, x_document_image: torch.Tensor, x_document_text: torch.Tensor, x_document_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_output = self.query_tower(
            x_query_context, x_query_image, x_query_text, x_query_action
        )
        document_output = self.document_tower(
            x_document_context, x_document_image, x_document_text, x_document_action
        )
        return query_output, document_output
