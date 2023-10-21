from llama_index.schema import Document
from typing import Union, Dict

import logging
import os
import sys

def convert_prompt_to_string(prompt) -> str:
    return prompt.format(**{v: v for v in prompt.template_vars})

def generate_query(profile: str, scan: str):
    return "Patient Profile: {}\nScan ordered: {}".format(profile, scan)

def convert_doc_to_dict(doc: Union[Document, Dict]) -> Dict:
    if isinstance(doc, Document):
        json_doc = {
            "page_content": doc.text,
            "metadata": {
                "source": doc.metadata["file_name"],
                "page": doc.metadata["page_label"]
            }
            }
    elif isinstance(doc, Dict):
        json_doc = {
            "page_content": doc["text"],
            "metadata": {
                "source": doc["metadata"]["file_name"],
                "page": doc["metadata"]["page_label"]
            }
        }
    return json_doc

def get_experiment_logs(description: str, log_folder: str):
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

