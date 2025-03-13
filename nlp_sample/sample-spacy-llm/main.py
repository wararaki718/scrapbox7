from pathlib import Path

from spacy_llm.util import assemble


def main() -> None:
    config_path = Path("./config.cfg")
    nlp = assemble(config_path=config_path)
    print("model loaded!")
    print()

    text = "Jack and Jill rode up the hill in Les Deux Alpes"
    print(f"sample: {text}")
    print()

    doc = nlp(text)
    print("entity")
    print([(ent.text, ent.label_) for ent in doc.ents])
    print()

    print("cats")
    print(doc.cats)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
