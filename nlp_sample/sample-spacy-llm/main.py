from pathlib import Path

from spacy_llm.util import assemble
from spacy.tokens.doc import Doc


def main() -> None:
    config_path = Path("./config.cfg")
    nlp = assemble(config_path=config_path)
    print("model loaded!")
    print()

    #text = "Jack and Jill rode up the hill in Les Deux Alpes"
    text = "Matthew and Maria went to Japan to visit the Nintendo headquarters"
    print(f"sample: {text}")
    print()

    doc = nlp(text)
    print(doc)
    print(type(doc))
    print()

    print(doc.vocab)
    print()

    print("entity")
    print([(ent.text, ent.label_) for ent in doc.ents])
    print()

    print("DONE")


if __name__ == "__main__":
    main()
