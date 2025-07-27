import logging

class ColoredFormatter(logging.Formatter):
    """
    A logging formatter that outputs colored log messages
    based on the log level.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # blue
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        """
        Format the specified record as text.
        """
        # Get the color corresponding to the log level. If not, use RESET
        color = self.COLORS.get(record.levelname, self.RESET)
        # Call the parent class to format the log message
        message = super().format(record)

        return f"{color}{message}{self.RESET}"

class SmartLogger:
    """Wrapper around logger with enhanced formatting support"""
    
    def __init__(self, logger):
        self._logger = logger
    
    def debug(self, msg, *args, **kwargs):
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        if self._logger.isEnabledFor(logging.CRITICAL):
            self._logger.critical(msg, *args, **kwargs)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate logging by not propagating to parent loggers
    logger.propagate = False

    # Avoid adding handlers repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return SmartLogger(logger)
