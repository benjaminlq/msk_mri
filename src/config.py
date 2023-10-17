"""Config file
"""
import logging
import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
EMB_DIR = os.path.join(DATA_DIR, "emb_store")
KG_DIR = os.path.join(DATA_DIR, "kg_store")

LOGGER = logging.getLogger(__name__)

stream_handler = logging.StreamHandler(sys.stdout)
log_folder = os.path.join(MAIN_DIR, "log")
if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok=True)

file_handler = logging.FileHandler(filename=os.path.join(log_folder, "logfile.log"))

formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)