from llama_index.schema import  NodeWithScore, TextNode

from typing import Union, Dict, List, Callable
import tiktoken
import re

def convert_prompt_to_string(prompt) -> str:
    return prompt.format(**{v: v for v in prompt.template_vars})

def generate_query(profile: str, scan: str):
    return "Patient Profile: {}\nScan ordered: {}".format(profile, scan)

def count_tokens(
    texts: Union[str, TextNode, NodeWithScore ,List],
    tokenizer: Callable = tiktoken.encoding_for_model("gpt-3.5-turbo")
):
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
):
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
    placeholders = re.findall(pattern = r"{([A-Za-z0-9_-]+)}", string=template)
    if isinstance(input_text, str):
        assert len(placeholders) == 1, "Must Provide a single placeholder when input_text is string."
        placeholder = placeholders[0]
        return template.format(**{placeholder:input_text})
    
    assert len(input_text) == len(placeholders)
    for key in input_text.keys():
        assert key in placeholders, f"{key} not present in template."
    
    return template.format(**input_text)