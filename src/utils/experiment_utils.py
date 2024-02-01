"""Utility Functions for running experiments
"""

import os
import sys
import logging
from logging import Logger

def get_experiment_logs(description: str, log_folder: str) -> Logger:
    """Generate logger. By default will log both to file and to terminal.

    Args:
        description (str): Description of the experiment
        log_folder (str): Folder containing the log file

    Returns:
        Logger: logger instance
    """
    logger = logging.getLogger(description)

    stream_handler = logging.StreamHandler(sys.stdout)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    file_handler = logging.FileHandler(filename=os.path.join(log_folder, "logfile.log"))

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger