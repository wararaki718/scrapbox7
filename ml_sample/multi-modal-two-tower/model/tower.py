import torch

from config import DocumentTowerConfig, QueryTowerConfig


class QueryTower(torch.nn.Module):
    def __init__(self, config: QueryTowerConfig) -> None:
        super().__init__()
        self._text_modal = torch.nn.Linear(config.text_input_dim, config.text_output_dim)
        self._image_modal = torch.nn.Linear(config.image_input_dim, config.image_output_dim)
        self._context_modal = torch.nn.Linear(config.context_input_dim, config.context_output_dim)
        self._actions_modal = torch.nn.Linear(config.actions_input_dim, config.actions_output_dim)
        self._model = torch.nn.Sequential(*[
            torch.nn.Linear(
                config.text_output_dim + config.image_output_dim + config.context_output_dim + config.actions_output_dim,
                config.output_dim,
            ),
            torch.nn.Dropout(p=config.dropout),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x_text: torch.Tensor, x_image: torch.Tensor, x_context: torch.Tensor, x_actions: torch.Tensor) -> torch.Tensor:
        x_text = self._text_modal(x_text)
        x_image = self._image_modal(x_image)
        x_context = self._context_modal(x_context)
        x_actions = self._actions_modal(x_actions)
        x = torch.cat([x_text, x_image, x_context, x_actions], dim=1)
        return self._model(x)


class DocumentTower(torch.nn.Module):
    def __init__(self, config: DocumentTowerConfig) -> None:
        super().__init__()
        self._text_modal = torch.nn.Linear(config.text_input_dim, config.text_output_dim)
        self._image_modal = torch.nn.Linear(config.image_input_dim, config.image_output_dim)
        self._context_modal = torch.nn.Linear(config.context_input_dim, config.context_output_dim)
        self._actions_modal = torch.nn.Linear(config.actions_input_dim, config.actions_output_dim)
        self._model = torch.nn.Sequential(*[
            torch.nn.Linear(
                config.text_output_dim + config.image_output_dim + config.context_output_dim + config.actions_output_dim,
                config.output_dim,
            ),
            torch.nn.Dropout(p=config.dropout),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x_text: torch.Tensor, x_image: torch.Tensor, x_context: torch.Tensor, x_actions: torch.Tensor) -> torch.Tensor:
        x_text = self._text_modal(x_text)
        x_image = self._image_modal(x_image)
        x_context = self._context_modal(x_context)
        x_actions = self._actions_modal(x_actions)
        x = torch.cat([x_text, x_image, x_context, x_actions], dim=1)
        return self._model(x)


class TwoTowerModel(torch.nn.Module):
    def __init__(self, document_tower_config: DocumentTowerConfig, query_tower_config: QueryTowerConfig) -> None:
        super().__init__()
        self.query_tower = QueryTower(query_tower_config)
        self.doc_tower = DocumentTower(document_tower_config)

    def forward(self, query_features: torch.Tensor, document_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_output = self.query_tower(query_features)
        document_output = self.doc_tower(document_features)
        return query_output, document_output
