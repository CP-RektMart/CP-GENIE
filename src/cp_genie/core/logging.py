import logging
import sys
from loguru import logger

def configure_logging(level: str = "INFO") -> None:
    """
    Replace the stdlib root handler with Loguru and forward all stdlib logs.
    """
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.root.setLevel(level)

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelname, record.getMessage())

    logging.root.addHandler(_InterceptHandler())
    
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}:{function}:{line}</cyan> - "
               "<level>{message}</level>",
    )
