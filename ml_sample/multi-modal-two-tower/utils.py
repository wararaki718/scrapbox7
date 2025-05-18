import torch


def get_text_features(n_data: int) -> torch.Tensor:
    return torch.randn(n_data, 128)

def get_image_features(n_data: int) -> torch.Tensor:
    return torch.randn(n_data, 64)

def get_context_features(n_data: int) -> torch.Tensor:
    return torch.randn(n_data, 32)

def get_actions_features(n_data: int) -> torch.Tensor:
    return torch.randn(n_data, 64)


def get_query_modalities(n_data: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_text = get_text_features(n_data)
    x_image = get_image_features(n_data)
    x_context = get_context_features(n_data)
    x_actions = get_actions_features(n_data)
    return x_text, x_image, x_context, x_actions


def get_document_modalities(n_data: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_text = get_text_features(n_data)
    x_image = get_image_features(n_data)
    x_context = get_context_features(n_data)
    x_actions = get_actions_features(n_data)
    return x_text, x_image, x_context, x_actions
