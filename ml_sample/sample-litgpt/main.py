from litgpt import LLM


def main() -> None:
    llm = LLM.load("microsoft/Phi-3.5-mini-instruct")
    text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
    print("output:")
    print(text)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
