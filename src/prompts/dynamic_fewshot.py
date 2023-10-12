from examples import examples
from config import MAIN_DIR
import os
import json
from typing import List

from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.embeddings import OpenAIEmbeddings

system_prompt = """
You are an honest medical clinician. If you do not know an answer, just say "I dont know", do not make up an answer.
======================
TASK
Your role is to refer to the reference information given under CONTEXT, analyse PATIENT PROFILE and recommend if the patient needs an MRI for based on their medication examination.

======================
CONTEXT:
{}

======================
OUTPUT INSTRUCTIONS
Your output should contain an answer and an explanation on whether an MRI should be ordered for the patient.
Answer must be one of [YES, NO]
Explanation must explain why an MRI should be ordered for the patient.
======================
EXAMPLES

======================
"""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "PATIENT PROFILE: {input}"),
        ("ai", "{output}")
    ]
)

def get_prompt(
    embeddings,
    vectorstore: VectorStore = FAISS,
    k: int = 2,
    examples: List = examples   
) -> ChatPromptTemplate:
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples = examples,
        embeddings = embeddings,
        vectorstore_cls = vectorstore,
        k = k    
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=example_prompt,
        example_selector=example_selector,
    )
    
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "PATIENT PROFILE: {input}")
        ]
    )
    
    return final_prompt

if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    
    embs = OpenAIEmbeddings(openai_api_key=api_keys["OPENAI_API_KEY"])
    final_prompt = get_prompt(embs)
    print(final_prompt.format(input = "non-specific patient but already under maintenance for 7 weeks with no improvement"))
    