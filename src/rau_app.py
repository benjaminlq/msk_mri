import os
import json
import sys
import glob
import openai
from llama_index import (
    VectorStoreIndex,
    LLMPredictor,
    SimpleDirectoryReader,
    ServiceContext,
    load_index_from_storage
)
import streamlit as st
from langchain.chat_models import ChatOpenAI
from config import DATA_DIR, MAIN_DIR

# Set the OpenAI API key. If unsure about obtaining the API key, refer to https://platform.openai.com/account/api-keys for more information.
# For estimating costs associated with index creation and chatbot usage, please visit: https://openai.com/pricing
with open(os.path.join(MAIN_DIR, "auth", "api_keys.json")) as f:
    api_keys = json.load(f)

openai.api_key = api_keys["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]

# Set the folder path for ACR data and the rebuild index flag. This script doesn't include
# importing ACR guidelines or a predefined index due to license restrictions on ACR guidelines.
# For more information, refer to the guidelines on the ACR website.

FOLDER_PATH = os.path.join(MAIN_DIR, "data" , "emb_store", "simple", "openai_512_20")
REBUILD_INDEX = False

def get_combined_chatbot_response(input_text):
    # Format the input text
    input_text = f'Case: {input_text} Question: Is imaging for this Usually Appropriate and if yes, state precisely only what imaging modality is the most Appropriate and if contrast agent is needed, do not mention "May Be Appropriate" or "Usually Not Appropriate".)'
    
    # Initialize the LLMPredictor with the ChatOpenAI model used in accGPT
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=512, request_timeout=120))
    
    # Create a service context for the LLMPredictor
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

    # Load the index from disk
    from llama_index.vector_stores import SimpleVectorStore
    from llama_index.storage import StorageContext
    vector_store = SimpleVectorStore.from_persist_dir(FOLDER_PATH)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=FOLDER_PATH)
    index = load_index_from_storage(storage_context=storage_context, index_id="msk-mri")

    # Define the messages to be sent to the models without index
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."},
        {"role": "user", "content": input_text},
    ]

    # accGPT approach: Query the index and get the response from Top 3 Text nodes using GPT-3.5-Turbo 
    query_engine = index.as_query_engine(
        service_context=service_context, 
        response_mode="compact",
        similarity_top_k=3
    )
    response = query_engine.query(input_text)
    output_accGPT = response.response.replace('\n', '\\n')

    # Get the response from GPT-3.5-Turbo
    response35 = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, timeout=120)
    output35 = response35.choices[0].message['content'].replace('\n', '\\n')
    
    # Get the response from GPT-4
    response4 = openai.ChatCompletion.create(model="gpt-4", messages=messages, timeout=120)
    output4 = response4.choices[0].message['content'].replace('\n', '\\n')
    
    # Combine the responses from all models
    answer = f"accGPT: {output_accGPT}\n\nGPT 3.5-Turbo: {output35}\n\nGPT 4: {output4}"

    return answer


# def launch_interface(chatbot_function):
#     iface = gr.Interface(
#         fn=get_combined_chatbot_response,
#         inputs=[gr.Textbox(lines=7, label="Enter your case")],
#         outputs=gr.Textbox(lines=7, label="Imaging Recommendations"),
#         title="AI Chatbots for Imaging Recommendations Aligned with ACR Guidelines"
#     )

#     # Launch the interface
#     iface.launch(share=True, debug=False)

# def main():
#     launch_interface(get_combined_chatbot_response)

# if __name__ == "__main__":
#     main()

st.title("AI Chatbots for Imaging Recommendations Aligned with ACR Guidelines")

convo = st.empty()
answer_container = st.empty()
spinner = st.empty()
query = st.empty()
    
with st.form("query"):

    patient_case = st.text_input("Enter your case: ", key="case")
    
    submitted = st.form_submit_button("Get AI Recommendation")
    if submitted:
        with convo.container():
            with spinner.container():
                with st.spinner(text="Generating guidelines for this patient. Please wait."):
                    response = get_combined_chatbot_response(patient_case)

            with answer_container.container():
                st.write("GPT Answer:")
                st.write(response)