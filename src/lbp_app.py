"""Streamlit App File
"""
from prompt import get_prompt

from typing import Optional

import streamlit as st
import logging

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
        llm=ChatOpenAI(temperature = 0, model_name=llm_type, max_tokens = 256,
                       openai_api_key=st.session_state.oai_api_key),
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


def handler_verify_key():
    """Function to verify whether input OpenAI API Key is working."""
    oai_api_key = st.session_state.open_ai_key_input
    try:
        model_list = [
            model_info["id"] for model_info in Model.list(api_key=oai_api_key)["data"]
        ]
        st.session_state.model_list = model_list
        st.session_state.oai_api_key = oai_api_key

    except Exception as e:
        with openai_key_container:
            st.error(f"{e}")
        logging.error(f"{e}")

### Enter Settings Here
emb_type = "text-embedding-ada-002"
k = 2
verbose = False

### Streamlit
st.set_page_config("Physician Medical Assistant", layout="wide")
st.title("AI assistant for MRI Order Recommendation for low back pain patients")

openai_key_container = st.container()

need_api_key_msg = ("Welcome! This app is a AI assistant that provides the recommendations"
                    "on whether a patient with low back pain should be sent for MRI scan" 
                   "It is powered by OpenAI's text models: gpt-3.5-turbo and gpt-4 To get started, "
                   "simply enter your OpenAI API Key below.")

helper_api_key_prompt = (
    "The model comparison tool works best with pay-as-you-go API keys. "
    "For more information on OpenAI API rate limits, check "
    "[this link](https://platform.openai.com/docs/guides/rate-limits/overview).\n\n"
    "- Don't have an API key? No worries! "
    "Create one [here](https://platform.openai.com/account/api-keys).\n"
    "- Want to upgrade your free-trial API key? Just enter your billing "
    "information [here](https://platform.openai.com/account/billing/overview)."
    )
helper_api_key_placeholder = "Paste your OpenAI API key here (sk-...)"

if "oai_api_key" not in st.session_state:
    st.write(need_api_key_msg)
    col1, col2 = st.columns([6, 4])
    col1.text_input(
        label="Enter OpenAI API Key",
        key="open_ai_key_input",
        type="password",
        autocomplete="current-password",
        on_change=handler_verify_key,
        placeholder=helper_api_key_placeholder,
        help=helper_api_key_prompt,
    )
    with openai_key_container:
        st.empty()
        st.write("---")
else:
    if "gpt-4" in st.session_state.model_list:
        llm_models = ["gpt-4", "gpt-3.5-turbo"]
    else:
        llm_models = ["gpt-3.5-turbo"]

    with st.sidebar:
        st.title("OpenAI Settings")
        llm_type = st.radio("LLM", llm_models)

    if emb_type == "text-embedding-ada-002":
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.oai_api_key)
    prompt = get_prompt(embeddings, vectorstore=FAISS, k=k)
    qa_chain = initialize_chain(_prompt_template=prompt, llm_type=llm_type, verbose=verbose)

    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []
    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []

    welcome_msg = ("This is an AI assistant that at advises on order of MRI Scan for patient with Low Back Pain"
                   "based on ACE Clinical guidelines. Please enter a short description of your patient case. For example:\n"
                   "1/ Patient with non-specific low back pain who has been under conservative maintenance for 6 weeks but symptoms do not subside.\n"
                   "2/ Patient with lower back pain and bilateral lower limb symptoms or signs\n"
                   )
    message(welcome_msg)

    convo = st.empty()
    query = st.empty()
    spinner = st.empty()

    # with conversation history
    # with convo.container():
    #     with query:
    #         user_query = get_text()
    #     if user_query:
    #         response = retriever(user_query)
    #         st.session_state["past"].append(user_query)
    #         st.session_state["generated"].append(response)
    #     if st.session_state["generated"]:
    #         for i in range(len(st.session_state["generated"])):
    #             message(st.session_state["past"][i], is_user=True)
    #             message(st.session_state["generated"][i])

    if "temp" not in st.session_state:
        st.session_state["temp"] = ""

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