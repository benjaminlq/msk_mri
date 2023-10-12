"""Config file
"""
import logging
import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
EMB_DIR = os.path.join(DATA_DIR, "emb_store")
KG_DIR = os.path.join(DATA_DIR, "kg_store")
