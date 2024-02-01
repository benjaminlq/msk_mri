"""Utility Functions for prompts generation
"""

from llama_index.schema import  NodeWithScore, TextNode

from llama_index.prompts import PromptTemplate
from typing import Union, Dict, List, Callable, Tuple
import tiktoken
import re

def convert_prompt_to_string(prompt: PromptTemplate) -> str:
    """Convert prompt template to string

    Args:
        prompt (PromptTemplate): Prompt Template (LlamaIndex format)

    Returns:
        str: String Prompt
    """
    return prompt.format(**{v: v for v in prompt.template_vars})

def generate_query(profile: str, scan: str) -> str:
    """Generate query provided patient profile and scan order

    Args:
        profile (str): Patient profile
        scan (str): Scan order

    Returns:
        str: Final Query
    """
    return "Patient Profile: {}\nScan ordered: {}".format(profile, scan)

def count_tokens(
    texts: Union[str, TextNode, NodeWithScore ,List],
    tokenizer: Callable = tiktoken.encoding_for_model("gpt-3.5-turbo")
):
    """Count total number of tokens of documents

    Args:
        texts (Union[str, TextNode, NodeWithScore ,List]): Documents or List of Documents
        tokenizer (Callable, optional): Tokenizer encoding function. Defaults to tiktoken.encoding_for_model("gpt-3.5-turbo").

    Returns:
        _type_: _description_
    """
    token_counter = 0
    if not isinstance(texts, List):
        texts = [texts]
    for text in texts:
        node = text.node if isinstance(text, NodeWithScore) else text       
        token_counter += len(tokenizer.encode(node.text))
    return token_counter

def remove_final_sentence(
    text: str,
    return_final_sentence: bool = False
) -> Union[str, Tuple[str, str]]:
    """Remove final sentence from a paragraph text

    Args:
        text (str): original_text
        return_final_sentence (bool, optional): Whether to return final sentence in the final output. Defaults to False.

    Returns:
        Union[str, Tuple[str, str]]: Text with final sentence removed.
    """
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    sentence_list = text.split(".")
    previous_text = ".".join(sentence_list[:-1])
    final_sentence = sentence_list[-1]
    return (previous_text, final_sentence) if return_final_sentence else previous_text

def query_wrapper(
    template: str, 
    input_text: Union[str, Dict[str, str]]
) -> str:
    """Works similar to PromptTemplate

    Args:
        template (str): String template. Placeholders are curly brackets {}
        input_text (Union[str, Dict[str, str]]): Input to place holders. If a single string, then template must have a single placeholder.

    Returns:
        str: String prompt
    """
    placeholders = re.findall(pattern = r"{([A-Za-z0-9_-]+)}", string=template)
    if isinstance(input_text, str):
        assert len(placeholders) == 1, "Must Provide a single placeholder when input_text is string."
        placeholder = placeholders[0]
        return template.format(**{placeholder:input_text})
    
    assert len(input_text) == len(placeholders)
    for key in input_text.keys():
        assert key in placeholders, f"{key} not present in template."
    
    return template.format(**input_text)