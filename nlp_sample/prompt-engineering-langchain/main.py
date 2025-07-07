from langchain_community.llms.llamacpp import LlamaCpp

from chains.single_prompt import chain_single_prompt
from chains.multi_prompt import chain_multi_prompt
from chains.memory_prompt import chain_memory_prompt, chain_window_memory_prompt
from chains.summary_prompt import chain_summary_prompt


def main() -> None:
    llm = LlamaCpp(
        model_path="models/Phi-3-mini-4k-instruct-fp16.gguf",
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False,
    )

    # single prompt
    text = "Hi! my name is Maarten. What is 1 + 1?"
    response = chain_single_prompt(text, llm)
    print(f"single prompt: {response}")
    print()

    # multi prompt
    text = "a girl that lost her mother"
    response = chain_multi_prompt(text, llm)
    print(f"multi prompt: {response}")
    print()

    # memory prompt
    texts = [
        "Hi! my name is Maarten. What is 1 + 1?",
        "What is my name?",
    ]
    responses = chain_memory_prompt(texts, llm)
    for i, response in enumerate(responses):
        print(f"memory prompt {i + 1}: {response}")
        print()
    
    # window memory prompt
    texts = [
        "Hi! my name is Maarten and I am 33 years old. What is 1 + 1?",
        "What is 3 + 3?",
        "What is my name?",
        "What is my age?",
    ]
    responses = chain_window_memory_prompt(texts, llm)
    for i, response in enumerate(responses):
        print(f"window memory prompt {i + 1}: {response}")
        print()
    
    # summary prompt
    texts = [
        "Hi! my name is Maarten. What is 1 + 1?",
        "What is my name?",
        "What was the first question I asked?",
    ]
    responses = chain_summary_prompt(texts, llm)
    for i, response in enumerate(responses):
        print(f"summary prompt {i + 1}: {response}")
        print()

    print("DONE")


if __name__ == "__main__":
    main()
