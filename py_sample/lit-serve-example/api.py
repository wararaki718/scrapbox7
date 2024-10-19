import torch
from litserve import LitAPI


class SimpleAPI(LitAPI):
    def setup(self, device: str) -> None:
        self._model1 = lambda x: x**2
        self._model2 = lambda x: x**3

    def decode_request(self, request: dict) -> int:
        print("input:")
        print(request["input"])
        print(type(request["input"]))
        return request["input"]

    def encode_response(self, output: int) -> dict:
        print("output:")
        print(output)
        print(type(output))
        return {"output": output}

    def predict(self, x: torch.Tensor) -> dict:
        squared = self._model1(x)
        cubed = self._model2(x)
        output = squared + cubed
        return output
