import torch

from config import DocumentTowerConfig
from .encoder import Encoder
from .modalities import ContextModality, ImageModality, TextModality, ActionModality


class DocumentTower(torch.nn.Module):
    def __init__(self, config: DocumentTowerConfig) -> None:
        super().__init__()
        # modalities
        self._context_modality = ContextModality(config.context_input_dim, config.context_output_dim)
        self._text_modality = TextModality(config.text_input_dim, config.text_output_dim)
        self._image_modality = ImageModality(config.image_input_dim, config.image_output_dim)
        self._action_modality = ActionModality(config.action_input_dim, config.action_output_dim)

        # encoder
        self._encoder = Encoder(
            input_dim=config.context_output_dim + config.text_output_dim + config.image_output_dim + config.action_output_dim,
            output_dim=config.output_dim
        )

    def forward(self, x_context: torch.Tensor, x_image: torch.Tensor, x_text: torch.Tensor, x_action: torch.Tensor) -> torch.Tensor:
        x_context = self._context_modality(x_context)
        x_text = self._text_modality(x_text)
        x_image = self._image_modality(x_image)
        x_action = self._action_modality(x_action)
        x = torch.cat([x_text, x_image, x_context, x_action], dim=1)
        return self._encoder(x)
