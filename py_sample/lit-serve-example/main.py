from litserve import LitServer

from api import SimpleAPI


def main() -> None:
    api = SimpleAPI()
    server = LitServer(api)
    server.run(port=8000)


if __name__ == "__main__":
    main()
