"""Utility Functions for retrieval
"""

from llama_index.embeddings.base import BaseEmbedding
from config import GUIDELINES

from textdistance import levenshtein
from typing import Union, List, Literal, Sequence, Tuple
import numpy as np

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