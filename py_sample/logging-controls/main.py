import logging

def show() -> None:
    logging.debug("debug")
    logging.info("info")
    logging.warning("warning")
    logging.error("error")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    show()
    print()

    print("disable info:")
    logging.disable(level=logging.INFO)
    show()
    logging.disable(level=logging.NOTSET)
    print()

    print("disable warning:")
    logging.disable(level=logging.WARNING)
    show()
    logging.disable(level=logging.NOTSET)

    print("DONE")


if __name__ == "__main__":
    main()
