from litgpt import LLM


def main() -> None:
    llm = LLM.load("EleutherAI/pythia-14m")
    text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
    print(text)

    print("DONE")


if __name__ == "__main__":
    main()
