import os
import sys
import time
import pprint
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
class Logger(object):

    def __init__(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)

        # show on
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()

def create_logger(args, info = None):
    timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
    log_dir = config.LOG_DIR
    if info:
        logger_name = os.path.join(log_dir, f"{current_time_str}_{info}.log")
    else:
        logger_name = os.path.join(log_dir, f"{current_time_str}_{args.dataset}_{args.llm}.log")
    logger = Logger(logger_name)

    logger.info(f"my pid: {os.getpid()}")
    formatted_args = pprint.pformat(args)
    logger.info(formatted_args)

    return logger