import torch


def generate_features(n_data: int, n_features: int) -> torch.Tensor:
    X = torch.randn(n_data, n_features)
    return X


def generate_labels(n_data: int) -> torch.Tensor:
    y = torch.randint(0, 2, (n_data,))
    return y


def get_data(
    n_data: int=100,
    n_context: int=10,
    n_image: int=20,
    n_text: int=30,
    n_action: int=10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_context = generate_features(n_data, n_context)
    X_image = generate_features(n_data, n_image)
    X_text = generate_features(n_data, n_text)
    X_action = generate_features(n_data, n_action)
    y = generate_labels(n_data)

    return X_context, X_image, X_text, X_action, y
