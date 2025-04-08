import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_output_info_log() -> None:
    logger.info("this is info")


def sample_raise_exception() -> None:
    raise Exception("this is an exception")


def sample_raise_exception_and_error_log() -> None:
    logger.error("this is an error")
    raise Exception("this is an exception")
