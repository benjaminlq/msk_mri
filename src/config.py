"""Config file
"""
import logging
import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
DOCUMENT_DIR = os.path.join(MAIN_DIR, "data", "document_sources")
EMB_DIR = os.path.join(DATA_DIR, "emb_store")
KG_DIR = os.path.join(DATA_DIR, "kg_store")

EXCLUDE_DICT = os.path.join(DATA_DIR, "exclude_pages.json")

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

GUIDELINES = [
    'acute hand and wrist trauma',
    'acute hip pain suspected fracture',
    'acute trauma to ankle',
    'acute trauma to foot',
    'acute trauma to knee',
    'aggressive primary msk tumour staging and surveillance',
    'cervical neck pain radiculopathy',
    'chroni extremity joint pain inflammatory arthritis',
    'chronic ankle pain',
    'chronic elbow pain',
    'chronic foot pain',
    'chronic hand and wrist pain',
    'chronic hip pain',
    'chronic knee pain',
    'chronic shoulder pain',
    'imaging after shoulder arthroplasty',
    'imaging after total hip arthroplasty',
    'imaging after total knee arthroplasty',
    'inflammatory back pain',
    'low back pain',
    'management of vertebral compression fracture',
    'myelopathy',
    'osteonecrosis',
    'primary bone tumours',
    'shoulder pain traumatic',
    'soft tissue mass',
    'stress fracture including sacrum',
    'suspected om foot in dm',
    'suspected om septic arthritis soft tissue infection',
    'suspected spine infection',
    'suspected spine trauma'
 ]