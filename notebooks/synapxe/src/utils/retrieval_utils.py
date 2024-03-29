"""Utility Functions for retrieval
"""

from llama_index.embeddings.base import BaseEmbedding
from config import GUIDELINES
from langchain.chains import LLMChain
from llama_index.embeddings.base import BaseEmbedding
from textdistance import levenshtein
from typing import Union, List, Literal, Sequence, Tuple
from typing import Union, List, Literal, Sequence, Tuple
import numpy as np
import re

def calculate_emb_distance(
    emb1: List[float],
    emb2: List[float],
    dist_type: Literal["l2", "ip", "cosine", "neg_exp_l2"] = "l2"
)->float:
    """Calculate the embedding distance between 2 embedding vectors

    Args:
        emb1 (List[float]): Embedding Vector 1
        emb2 (List[float]): Embedding Vector 1
        dist_type (Literal[l2, ip, cosine, neg_exp_l2], optional): Distance type. Defaults to "l2".

    Returns:
        float: Distance
    """
    assert len(emb1) == len(emb2), "Length of embedding vectors must match"
    if dist_type == "l2":
        return np.square(np.linalg.norm(np.array(emb1) - np.array(emb2)))
    elif dist_type == "ip":
        return 1 - np.dot(emb1, emb2)
    elif dist_type == "cosine":
        cosine_similarity = np.dot(emb1, emb2)/(np.norm(emb1)*np.norm(emb2))
        return 1 - cosine_similarity
    elif dist_type == "neg_exp_l2":
        return np.exp(-np.square(np.linalg.norm(np.array(emb1) - np.array(emb2))))
    else:
        raise ValueError("Invalid distance type")
    
def calculate_string_distance(
    str1: str,
    str2: Union[str, Sequence[str]],
    embeddings: BaseEmbedding,
    dist_type: Literal["l2", "ip", "cosine", "neg_exp_l2"] = "l2"
) -> Union[float, List[float]]:
    """Calculate string distance using vector embeddings

    Args:
        str1 (str): String 1 (Query)
        str2 (Union[str, Sequence[str]]): String 2 or List of reference strings
        embeddings (BaseEmbedding): Embedding Model
        dist_type (Literal[l2, ip, cosine, neg_exp_l2], optional): _description_. Defaults to "l2".

    Returns:
        Union[float, List[float]]: Distance or list of distances between query string and reference string(s)
    """
    emb1 = embeddings.get_query_embedding(str1)
    if isinstance(str2, str):
        emb2 = embeddings.get_text_embedding(str2)
        return calculate_emb_distance(emb1, emb2, dist_type)
    else:
        emb2_list = embeddings.get_text_embedding_batch(str2)
        return [calculate_emb_distance(emb1, emb2) for emb2 in emb2_list]
    
def calculate_min_dist(
    input_str: str,
    text_list: List[str] = GUIDELINES,
    return_nearest_text: bool = False
) -> Union[str, Tuple[int, str]]:
    """Find the closest string from a list of reference strings to a query string.
    Use Levenstein distance to calculate distance between strings.

    Args:
        input_str (str): Query string
        text_list (List[str], optional): List of reference strings. Defaults to GUIDELINES.
        return_nearest_text (bool, optional): If True, return the minimum Levenstein distance. Defaults to False.

    Returns:
        Union[str, Tuple[int, str]]: Nearest reference string.
    """
    min_dist = float("inf")
    nearest_text = None

    for ref_text in text_list:
        dist = levenshtein.distance(input_str, ref_text)
        if dist < min_dist:
            min_dist = dist
            nearest_text = ref_text
    return (min_dist, nearest_text) if return_nearest_text else min_dist

def extract_guidelines(
    profile: str,
    extract_chain: LLMChain
) -> Tuple[str, List[str]]:
    """Function to extract relevant guidelines

    Args:
        profile (str): Input patient profile
        extract_chain (LLMChain): LLM call to extract relevant guidelines

    Returns:
        Tuple[str, List[str]]: profile, relevant_guidelines
    """
    extracted_response = extract_chain(profile)["text"]
    if extracted_response.endswith("."):
        extracted_response = extracted_response[:-1]

    pattern = r"1. Relevant information:([\S\s]+)2. Relevant guidelines:([\S\s]*)"

    profile, guidelines_str = re.findall(pattern, extracted_response)[0]
    guidelines_str = guidelines_str.replace("- ", "")
    guidelines_str = guidelines_str.strip()
    guidelines_str = guidelines_str.replace("\n", ", ")

    if not guidelines_str:
        relevant_guidelines = []
    else:
        regex_guidelines = re.findall(r"([A-Za-z ]+)", guidelines_str)
        relevant_guidelines = []
        for extracted_guideline in regex_guidelines:
            extracted_guideline = extracted_guideline.lower()
            min_dist, nearest_text = calculate_min_dist(extracted_guideline, GUIDELINES, True)
            if min_dist <= 1:
                extracted_guideline = nearest_text
                relevant_guidelines.append(extracted_guideline)
                
    return profile, relevant_guidelines