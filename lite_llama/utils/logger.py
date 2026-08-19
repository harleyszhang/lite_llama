import logging
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Legacy-style formatter: ``04-24 09:30:45 [I] model_executor.py:145 msg``.

    Level names are shortened to their one-letter form (``[I]``/``[W]``/``[E]``)
    and the record's file name *and line number* are included, matching the
    log output the original ``utils/logger.py`` produced.
    """

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # blue
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    LEVEL_SIM: ClassVar[dict[str, str]] = {
        "DEBUG": "[D]",
        "INFO": "[I]",
        "WARNING": "[W]",
        "ERROR": "[E]",
        "CRITICAL": "[C]",
    }
    RESET = "\033[0m"

    def format(self, record):
        """Shorten the level, colour the line, keep ``file.py:lineno``."""
        # Get the color corresponding to the log level. If not, use RESET
        color = self.COLORS.get(record.levelname, self.RESET)
        # Temporarily swap the level name for its [I]/[W]/[E] short form.
        original_level = record.levelname
        record.levelname = self.LEVEL_SIM.get(record.levelname, original_level)
        try:
            message = super().format(record)
        finally:
            record.levelname = original_level

        return f"{color}{message}{self.RESET}"


class SmartLogger:
    """Deprecated pass-through kept for API compatibility.

    Wrapping the logger used to swallow the caller's file/line — ``%(filename)s``
    and ``%(lineno)d`` resolved to this module instead of the code that logged.
    The methods below are kept so any external reference still works, but they
    delegate straight to the underlying logger.
    """

    def __init__(self, logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate logging by not propagating to parent loggers
    logger.propagate = False

    # Avoid adding handlers repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Return the bare logger: the extra wrapper layer would make
    # %(filename)s / %(lineno)d resolve to logger.py instead of the caller.
    # (logging.Logger.info already short-circuits on isEnabledFor, so the
    # wrapper added nothing but the wrong file name in the legacy format.)
    return logger
