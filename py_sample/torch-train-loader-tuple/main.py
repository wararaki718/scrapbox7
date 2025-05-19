import torch

from modality import MultiModalDataset
from utils import get_data


def main() -> None:
    X_context, X_image, X_text, X_action, y = get_data()
    dataset = MultiModalDataset(X_context, X_image, X_text, X_action, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    for i, batch in enumerate(dataloader):
        X_context_batch, X_image_batch, X_text_batch, X_action_batch, y_batch = batch
        print(f"## Batch {i + 1}:")
        print(f"X_context: {X_context_batch.shape}")
        print(f"X_image: {X_image_batch.shape}")
        print(f"X_text: {X_text_batch.shape}")
        print(f"X_action: {X_action_batch.shape}")
        print(f"y: {y_batch.shape}")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
