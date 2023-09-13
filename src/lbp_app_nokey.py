"""Streamlit App File
"""
from prompt import get_prompt

from typing import Optional

import streamlit as st
import os

from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexDictParser

from openai import Model
from streamlit_chat import message

### Functions
@st.cache_resource()
def initialize_chain(
    _prompt_template: ChatPromptTemplate,
    llm_type: str = "gpt-3.5-turbo",
    verbose: Optional[bool] = None,
):

    qa_chain = LLMChain(
        prompt=_prompt_template,
        llm=ChatOpenAI(temperature = 0, model_name=llm_type, max_tokens = 256),
        output_key="answer",
        # output_parser=RegexDictParser(
        #     output_key_to_format={"answer": "Answer", "explanation": "Explanation"}
        #     ),
        verbose=verbose
    )

    return qa_chain


def get_text():
    """Prompt users to input query.

    Returns:
        str: User's input query
    """
    st.text_input(
        "Enter patient case scenario below: ", key="query", on_change=clear_text
    )
    return st.session_state["temp"]


def clear_text():
    """This function helps to clear the previous text input from the input field.
    Temporary assign the input value to "temp" and clear "query" from session_state."""
    st.session_state["temp"] = st.session_state["query"]
    st.session_state["query"] = ""

### Enter Settings Here
emb_type = "text-embedding-ada-002"
k = 2
verbose = False

### Streamlit
st.set_page_config("Physician Medical Assistant", layout="wide")
st.title("AI assistant for MRI Order Recommendation for low back pain patients")

openai_key_container = st.container()

model_list = [
    model_info["id"] for model_info in Model.list(api_key=os.environ["OPENAI_API_KEY"])["data"]
]
        
if "gpt-4" in model_list:
    llm_models = ["gpt-4", "gpt-3.5-turbo"]
else:
    llm_models = ["gpt-3.5-turbo"]

with st.sidebar:
    st.title("OpenAI Settings")
    llm_type = st.radio("LLM", llm_models)

if emb_type == "text-embedding-ada-002":
    embeddings = OpenAIEmbeddings()
prompt = get_prompt(embeddings, vectorstore=FAISS, k=k)
qa_chain = initialize_chain(_prompt_template=prompt, llm_type=llm_type, verbose=verbose)

welcome_msg = ("This is an AI assistant that at advises on order of MRI Scan for patient with Low Back Pain"
                "based on ACE Clinical guidelines. Please enter a short description of your patient case. For example:\n"
                "1/ Patient with non-specific low back pain who has been under conservative maintenance for 6 weeks but symptoms do not subside.\n"
                "2/ Patient with lower back pain and bilateral lower limb symptoms or signs\n"
                )
message(welcome_msg)

convo = st.empty()
query = st.empty()
disclaimer = st.empty()
spinner = st.empty()

if "temp" not in st.session_state:
    st.session_state["temp"] = ""

with disclaimer.container():
    st.write(("DISCLAIMER: This application will send data to OpenAI API Server.\n"
              "Please DE-IDENTIFY your patient profile before asking the assistant."))

with query.container():
    user_query = get_text()

with convo.container():
    if user_query:
        message(user_query, is_user=True)
        with spinner.container():
            with st.spinner(text="Generating guidelines for this patient. Please wait."):
                response = qa_chain(user_query)["answer"]

        message(response)

# streamlit run src/scripts/lbp_app.py