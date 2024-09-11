import logging
import sys
from time import perf_counter

import datasets
import openvino

from loader import ModelLoader

logging.basicConfig(
    format="[ %(levelname)s ] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    model_name = "bert-base-uncased"
    model, tokenizer = ModelLoader.load(model_name)
    infer_request = openvino.AsyncInferQueue(model)
    logger.info("model loaded.")

    sst2 = datasets.load_dataset("glue", "sst2")
    sentences = sst2["validation"]["sentence"]
    logger.info("data loaded.")

    # warmup
    encoded_warm_up = dict(tokenizer("warm up sentence", return_tensors="np"))
    for _ in range(len(infer_request)):
        infer_request.start_async(encoded_warm_up)
    infer_request.wait_all()
    logger.info("warmup done.")

    # benchmark
    sum_seq_len = 0
    start = perf_counter()
    for sentence in sentences:
        encoded = dict(tokenizer(sentence, return_tensors="np"))
        sum_seq_len += next(iter(encoded.values())).size
        infer_request.start_async(encoded)
    infer_request.wait_all()
    end = perf_counter()

    duration = end - start
    logger.info(f"duration: {duration} sec")

    logger.info("DONE")


if __name__ == "__main__":
    main()
