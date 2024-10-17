import torch
import torch.nn as nn
import torch.nn.functional as F

from model import QueryEncoder, DocumentEncoder
from trainer import Trainer
from utils import load_dummy_data


def main() -> None:
    query_input_size = 20
    document_input_size = 15

    learning_rate = 0.2
    margin = 1.0
    num_epochs = 100

    query_encoder = QueryEncoder(query_input_size)
    document_encoder = DocumentEncoder(document_input_size)

    query_optimizer = torch.optim.Adam(query_encoder.parameters(), lr=learning_rate)
    document_optimizer = torch.optim.Adam(document_encoder.parameters(), lr=learning_rate)

    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=margin
    )
    trainer = Trainer(
        query_encoder,
        document_encoder,
        criterion,
        query_optimizer,
        document_optimizer,
    )
    data_loader = load_dummy_data(100, query_input_size, document_input_size)
    for epoch in range(1, num_epochs+1):
        trainer.train(data_loader, epoch)
    print("DONE")


if __name__ == "__main__":
    main()
