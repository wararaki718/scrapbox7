from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer, util


def main() -> None:
    dataset = load_dataset("jamescalam/ai-arxiv-chunked")
    chunks = dataset["train"]["chunk"]
    print(f"the number of chunks: {len(chunks)}")
    print()

    # bi-encoder
    bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    bi_encoder.max_seq_length = 256
    corpus_embeddings = bi_encoder.encode(
        chunks,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    print(corpus_embeddings.shape)
    print()

    print("bi-encoder (retrieve):")
    query = "what is rlhf?"
    top_k = 25
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    retrieval_corpus_ids = [hit['corpus_id'] for hit in hits]

    print("cross-encoder (reranking):")
    cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
    cross_inp = [[query, chunks[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    bge_corpus_ids = [hit['corpus_id'] for hit in hits]
    print()

    for i in range(top_k):
        print(f"Top {i+1} passage. Bi-encoder {retrieval_corpus_ids[i]}, Cross-encoder BGE {bge_corpus_ids[i]}")
    print("DONE")


if __name__ == "__main__":
    main()
