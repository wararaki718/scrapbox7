from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from prompt import get_prompt
from utils import get_texts


def main() -> None:
    llm = LlamaCpp(
        model_path="./model/Phi-3-mini-4k-instruct-fp16.gguf",
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False,
    )
    print("LLM loaded!")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("Embedding model loaded!")

    texts = get_texts()
    print(f"Number of texts: {len(texts)}")

    db = FAISS.from_texts(texts, embedding_model)
    print("Vector store created!")

    prompt = get_prompt()
    print("Prompt loaded!")

    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt
        },
        verbose=True,
    )
    print("RAG chain created!")

    query = "Income generated"
    result = rag.invoke(query)
    print(f"Query: {query}")
    print(f"Result: {result}")

    print("DONE")


if __name__ == "__main__":
    main()
