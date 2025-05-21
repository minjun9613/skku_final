# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_name=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger
