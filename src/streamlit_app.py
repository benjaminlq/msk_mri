"""Streamlit App File
"""
from typing import Optional

import streamlit as st
import os, json

from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexDictParser
from langchain.vectorstores.base import VectorStore
from langchain.document_transformers import LongContextReorder

from custom.lc_chains import ReOrderQARetrieval
from openai import Model
from streamlit_chat import message

import pandas as pd
import config

### Functions
@st.cache_resource()
def initialize_chain(
    llm_type: str = "gpt-4",
    k: int = 5,
    verbose: bool = False,
):
    # prompt = get_prompt(embeddings, vectorstore=FAISS, k=k) # Dynamic Few-Shot Prompts
    from prompts.zero_shot import PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE
    docsearch = FAISS.load_local(emb_store, embeddings=embeddings)
    qa_chain = ReOrderQARetrieval.from_chain_type(
        llm=ChatOpenAI(model_name=llm_type, temperature=0, max_tokens=512),
        retriever=docsearch.as_retriever(search_kwargs={"k": k}),
        reorder_fn = LongContextReorder(),
        chain_type="stuff",
        chain_type_kwargs=dict(
            document_variable_name = "context",
            prompt=prompt
        ),
        input_key="question",
        return_source_documents = True,
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

def handler_download_key():
    st.session_state.file = None
    with batch_status_container.container():
        st.write("")

### Enter Settings Here
emb_type = "text-embedding-ada-002"
emb_store = os.path.join(config.EMB_DIR, "faiss", "openai_1024_128")
k = 5
llm_type = "gpt-4"
verbose = False

if not os.environ.get("OPENAI_API_KEY"):
    with open(os.path.join(config.MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]

### Streamlit
st.set_page_config("Physician Medical Assistant", layout="wide")
st.title("AI assistant for MRI Order Recommendation for low back pain patients")

openai_key_container = st.container()

#### LLM Type ####
# model_list = [
#     model_info["id"] for model_info in Model.list(api_key=os.environ["OPENAI_API_KEY"])["data"]
# ]
        
# if "gpt-4" in model_list:
#     llm_models = ["gpt-4", "gpt-3.5-turbo"]
# else:
#     llm_models = ["gpt-3.5-turbo"]

# with st.sidebar:
#     st.title("OpenAI Settings")
#     llm_type = st.radio("LLM", llm_models)

if emb_type == "text-embedding-ada-002":
    embeddings = OpenAIEmbeddings()
    
qa_chain = initialize_chain(llm_type=llm_type, k=k, verbose=verbose)

welcome_msg = ("This is an AI assistant that at advises on the appropriateness of MRI Scan order "
               "for patient with Low Back and Spine Conditions based on ACE Clinical guidelines. "
               "Please enter a short description of your patient case and the type of scan ordered.\n\n"
               "For example:\n"
               "**Patient Profile**: Patient with non-specific low back pain who has been under conservative maintenance for 6 weeks but symptoms do not subside.\n"
               "**Scan ordered**: MRI lumbar spine with IV contrast"
                )

st.header("Interactive Chat")
message(welcome_msg)
disclaimer = st.empty()

convo = st.empty()
query = st.empty()
spinner = st.empty()
answer_container = st.empty()

if "temp" not in st.session_state:
    st.session_state["temp"] = ""

with disclaimer.container():
    st.write(("DISCLAIMER: This application will send data to OpenAI API Server.\n"
            "Please DE-IDENTIFY your patient profile before asking the assistant."))

with st.form("query"):

    patient_profile = st.text_input("Enter patient case scenario below: ", key="profile")
    scan_order = st.text_input("Enter Scan Order: ", key="scan")
    
    submitted = st.form_submit_button("Get AI Recommendation")
    if submitted:
        user_query = "**Patient Profile**:{}\n**Scan ordered**:{}".format(patient_profile, scan_order)

        with convo.container():
            # if user_query:
            #     message(user_query, is_user=True)
            #     with spinner.container():
            #         with st.spinner(text="Generating guidelines for this patient. Please wait."):
            #             response = qa_chain(user_query)

            #     message(response["result"])
            
            message(user_query, is_user=True)
            with spinner.container():
                with st.spinner(text="Generating guidelines for this patient. Please wait."):
                    response = qa_chain(user_query)

            with answer_container.container():
                message(response["result"])
            
with st.sidebar:
    st.header("Batch Process")
    st.write(
        (
            "Batch Process accepts .csv file in one of the following formats:\n"
            "1. A .csv file with 2 columns. 1st column is Patient Profile, 2nd column is MRI Scan Ordered. Column name can be arbitrary\n"
            "2. A .csv file with multiple columns. Must contain 2 columns with exact names 'Clinical File' for patient profile and 'MRI scan ordered' for MRI scan ordered."
            )
        )
    
    batch_container = st.empty()
    batch_status_container = st.empty()
    download_container = st.empty()

    with batch_container.container():
        with st.form("batch"):
            st.session_state.file = st.file_uploader(label="Please Upload a file", type="csv")
            batch_submitted = st.form_submit_button("Get Batch Result")
            if batch_submitted:
                if not st.session_state.file:
                    st.write("Please upload a file first!!!")
                else:
                    completion = 0
                    progress_text = "Operation in progress. Please wait. "
                    progress_bar = st.progress(completion, text=progress_text + "Completed: {:d} %".format(int(completion*100)))
                    
                    df = pd.read_csv(st.session_state.file)
                    
                    samples_no = len(df)
                    progress_step = 1.0 / samples_no
                    
                    if len(df.columns) == 2:
                        df.columns = ["Clinical File", "MRI scan ordered"]
                        
                    patient_profiles = df["Clinical File"]
                    scan_orders = df["MRI scan ordered"]
                    queries = ["**Patient Profile**:{}\n**Scan ordered**:{}".format(patient_profile, scan_order)
                            for patient_profile, scan_order in zip(patient_profiles, scan_orders)]
                    
                    answers = []

                    for idx, query in enumerate(queries):
                        response = qa_chain(query)
                        answers.append(response["result"])
                        completion += progress_step
                        progress_bar.progress(int(completion*100), text=progress_text + "Completed: {:d} %".format(int(completion*100)))
                    
                    df["GPT answer"] = answers
                    st.session_state.df = df                                
                    progress_bar.empty()
                    with batch_status_container.container():
                        st.write("Completed!!!")
    
    with download_container.container():  
        if "df" in st.session_state:
            st.download_button(
                label="Download data as CSV",
                data=st.session_state.df.to_csv().encode("utf-8"),
                file_name='output.csv',
                on_click=handler_download_key,
                mime='text/csv',
                )
            
# streamlit run src/app/streamlit_app.py