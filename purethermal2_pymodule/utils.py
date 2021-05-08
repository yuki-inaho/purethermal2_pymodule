from logging import getLogger, StreamHandler, Formatter, INFO


def get_logger_with_stdout(name: str):
    handler = StreamHandler()
    handler.setFormatter(Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"))
    logger = getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(INFO)
    return logger
