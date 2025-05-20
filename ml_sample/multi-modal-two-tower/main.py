from config import DocumentTowerConfig, QueryTowerConfig
from model.tower import TwoTowerModel
from model.modality import Modalities, MultiModalDataset
from utils import get_data


def main() -> None:
    # load config
    document_tower_config = DocumentTowerConfig()
    query_tower_config = QueryTowerConfig()

    # get modalities
    n_data = 100
    (
        X_document_context,
        X_document_image,
        X_document_text,
        X_document_action,
        y,
    ) = get_data(
        n_data=n_data,
        n_context=document_tower_config.context_input_dim,
        n_image=document_tower_config.image_input_dim,
        n_text=document_tower_config.text_input_dim,
        n_actions=document_tower_config.action_input_dim,
    )
    document_modalities = Modalities(
        X_context=X_document_context,
        X_image=X_document_image,
        X_text=X_document_text,
        X_action=X_document_action,
    )
    print("document modalities defined")

    (
        X_query_context,
        X_query_image,
        X_query_text,
        X_query_action,
        _,
    ) = get_data(
        n_data=n_data,
        n_context=query_tower_config.context_input_dim,
        n_image=query_tower_config.image_input_dim,
        n_text=query_tower_config.text_input_dim,
        n_actions=query_tower_config.action_input_dim,
    )
    query_modalities = Modalities(
        X_context=X_query_context,
        X_image=X_query_image,
        X_text=X_query_text,
        X_action=X_query_action,
    )
    print("query modalities defined")

    # create dataset
    dataset = MultiModalDataset(
        query_modalities=query_modalities,
        document_modalities=document_modalities,
        y=y,
    )
    print("dataset defined")

    # define model
    model = TwoTowerModel(document_tower_config, query_tower_config)
    print("model defined")
    
    print("DONE")


if __name__ == "__main__":
    main()
