
import os
import glob
import logging
from datetime import datetime

class ConsoleFormatter(logging.Formatter):
    """Custom formatter to colorize errors in console output."""
    def format(self, record):
        if record.levelno == logging.ERROR:
            return f"\033[91m{super().format(record)}\033[0m"  # Red for errors
        if record.levelno == logging.WARNING:
            return f"\033[33m{super().format(record)}\033[0m"  # Yellow for warnings
        if record.levelno == logging.DEBUG:
            return f"\033[95m{super().format(record)}\033[0m"  # Magenta for debug messages
        if record.levelno == logging.CRITICAL:
            return f"\033[91;1m{super().format(record)}\033[0m"  # Bold red for critical messages
        return super().format(record)


def setup_logger(level=logging.INFO, log_dir="src/logs", pretty=False):
    """
    Set up a logger that logs to both console and file.
    
    Input:
    - level: Logging level (e.g., logging.DEBUG, logging.INFO)
    - log_dir: Directory to save log file. Default is 'src/logs/'
    - pretty: If True, use colored output in console (default is False)
    
    Output:
    - logger: Configured logger object
    """    
    # define log file
    os.makedirs(log_dir, exist_ok=True)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'log_{start_time}.log')
    
    # create logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # prevent duplicate messages
    if not logger.handlers:
        
        # create console handler (print to console)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # create file handler (write to file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # define console format (message only)
        if pretty:
            formatter = ConsoleFormatter("%(message)s")  # colored output
        else:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # standard format
        console_handler.setFormatter(formatter)
        
        # define file format (timestamp - log level - message)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger