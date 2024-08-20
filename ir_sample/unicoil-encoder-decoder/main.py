from encoder import DenseContextEncoder, DenseQueryEncoder, SparseContextEncoder, SparseQueryEncoder


def main() -> None:
    context_dpr_model_name = "facebook/dpr-ctx_encoder-multiset-base"
    dense_context_encoder = DenseContextEncoder(context_dpr_model_name)

    query_dpr_model_name = "facebook/dpr-question_encoder-multiset-base"
    dense_query_encoder = DenseQueryEncoder(query_dpr_model_name)

    context_spar_model_name = "facebook/spar-marco-unicoil-lexmodel-context-encoder"
    sparse_context_encoder = SparseContextEncoder(context_spar_model_name)

    query_spar_model_name = "facebook/spar-marco-unicoil-lexmodel-query-encoder"
    sparse_query_encoder = SparseQueryEncoder(query_spar_model_name)
    
    query = "Where was Marie Curie born?"
    contexts = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    dense_query_embeddings = dense_query_encoder.encode(query)
    dense_context_embeddings = dense_context_encoder.encode(contexts)
    print(dense_query_embeddings.shape)
    print(dense_context_embeddings.shape)

    sparse_query_embeddings = sparse_query_encoder.encode(query)
    sparse_context_embeddings = sparse_context_encoder.encode(contexts)
    print(sparse_query_embeddings.shape)
    print(sparse_context_embeddings.shape)


if __name__ == "__main__":
    main()
