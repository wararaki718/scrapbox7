from config import DocumentTowerConfig, QueryTowerConfig
from model.tower import TwoTowerModel
from utils import get_document_modalities, get_query_modalities

def main() -> None:
    document_tower_config = DocumentTowerConfig()
    query_tower_config = QueryTowerConfig()

    model = TwoTowerModel(document_tower_config, query_tower_config)
    print("model defined")

    document_modalities = get_document_modalities(100)
    query_modalities = get_query_modalities(100)
    print("modalities defined")

    
    print("DONE")


if __name__ == "__main__":
    main()
