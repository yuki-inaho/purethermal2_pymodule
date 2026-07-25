import logging

from purethermal2_pymodule.utils import get_logger_with_stdout


def test_get_logger_with_stdout_sets_name_and_level():
    logger = get_logger_with_stdout("test-logger-name-and-level")
    try:
        assert logger.name == "test-logger-name-and-level"
        assert logger.level == logging.INFO
    finally:
        logger.handlers.clear()


def test_get_logger_with_stdout_attaches_stream_handler_with_formatter():
    logger = get_logger_with_stdout("test-logger-handler")
    try:
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert (
            handler.formatter._fmt
            == "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"
        )
    finally:
        logger.handlers.clear()
