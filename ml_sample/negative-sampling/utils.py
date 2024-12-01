import torch


def load_dummy_data(
    n_queries: int=20,
    n_documents: int=100,
    n_impressions: int=10,
    n_min_actions: int=2,
    n_features: int=256,
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    # create features
    X_queries = torch.randn((n_queries, n_features), dtype=torch.float32)
    X_documents = torch.randn((n_documents, n_features), dtype=torch.float32)

    # impression
    impressions = []
    for i in range(n_queries):
        # random documents
        document_ids: torch.Tensor = torch.randperm(n_documents)[:n_impressions]

        # random interaction
        actions: torch.Tensor = torch.zeros(n_impressions, dtype=int)
        positions = torch.randint(0, n_impressions, (n_min_actions,))
        actions[positions] = 1

        # impression
        impression = {"query_id": i, "document_ids": document_ids, "actions": actions}
        impressions.append(impression)

    # data_loader = DataLoader(
    #     TripletDataset(X_query, X_pos, X_neg),
    #     batch_size=5,
    #     shuffle=True,
    #     num_workers=2,
    #     persistent_workers=True,
    # )
    return impressions, X_queries, X_documents
