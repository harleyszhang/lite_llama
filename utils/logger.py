# -*- coding: UTF-8 -*-

import os
import sys
sys.path.append("..")
import logging
from utils.common import getTime
from utils.common import getProjectPath

# reload(sys)
# sys.setdefaultencoding('utf-8')

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

LEVEL_SIM = {
    'WARNING': '[W]',
    'INFO': '[I]',
    'DEBUG': '[D]',
    'CRITICAL': '[C]',
    'ERROR': '[E]'
}


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg, datefmt="%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            simple_ln = LEVEL_SIM.get(levelname)
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + simple_ln + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "%(asctime)s $RESET%(levelname)s %(filename)s$RESET:%(lineno)d %(message)s "
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.ERROR)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return


project_path = getProjectPath()


def loggerHandle():
    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


def logfileHandle(log_name="logs/common.log"):
    log_file = os.path.join(project_path, log_name)
    if not os.path.exists(os.path.join(project_path, 'logs')):
        os.makedirs(os.path.join(project_path, 'logs'))
    if not os.path.exists(log_file):
        os.mknod(log_file)
    logfile = logging.getLogger()
    logfile.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file, encoding='UTF-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d  %(message)s', datefmt="%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logfile.addHandler(handler)
    return logfile


log = loggerHandle()
logE = logfileHandle("logs/error.log")
logP = logfileHandle("logs/post.log")
logU = logfileHandle("logs/upload_data.log")

import time
if __name__ == '__main__':
    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug("\033[1;32mMessage Error\033[0m")
    logger.info("test")
    logger.warning("test")
    logger.error("test")
    time.sleep(10)
    logger.info("aaaaa")
#
#     logfile = logging.getLogger()
#     logfile.setLevel(logging.DEBUG)
#     handler = logging.FileHandler("Alibaba.log", encoding='UTF-8')
#     formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s')
#     handler.setFormatter(formatter)
#     logfile.addHandler(handler)
#     logfile.info("aaaaaaaaaaaaaaa")
