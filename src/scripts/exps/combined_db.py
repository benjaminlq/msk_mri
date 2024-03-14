"""Script to run combined db experiment
"""
import argparse
import json
import os
import pandas as pd
import tiktoken

from config import MAIN_DIR, DATA_DIR, ARTIFACT_DIR
from prompts.llama_index.rag_v1 import CHAT_PROMPT_TEMPLATE
from custom import CustomCombinedRetriever, CustomRetrieverQueryEngine
from datetime import datetime

from llama_index import ServiceContext, get_response_synthesizer
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from tqdm import tqdm
from utils import *

def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser()
    
    # Experiment Settings
    parser.add_argument(
        "--description", "-desc", type=str, help="Description of the experiment" 
    )
    parser.add_argument(
        "--n_iterations", "-n", type=int, default=1, help="Number of iterations to be repeated"
    )
    parser.add_argument(
        "--refine_profile", action='store_true', default=False, help="Whether refine the patient profile before query"
    )
    parser.add_argument(
        "--refine_llm", type=str, default= "gpt-4", help="LLM model to be used to refine patient profile"
    )
    parser.add_argument(
        "--dataset", type=str, default=os.path.join(DATA_DIR, "queries", "MSK LLM Fictitious Case Files Full - Reorder.csv"), help="path to query dataset"
    )
    # VectorDB & Retrieval settings
    parser.add_argument(
        "--embed_store", type=str, default="chroma", help="simple|chroma|faiss|pinecone|weavoate"
    )
    parser.add_argument(
        "--database", "-db", type=str, help="path to database folder",
    )
    parser.add_argument(
        "--index_name", "-i", type=str, help="Name of the index for vector db",
    )
    parser.add_argument(
        "--embed_model", "-e", type=str, default="openai", help="path to embedding model",
    )
    parser.add_argument(
        "--emb_size", "-d", type=int, default=1536, help="Dimension of embedding vector",
    )
    parser.add_argument(
        "--chunk_size", "-s", type=int, default=512, help="Chunk size of the embedded texts"
    )
    parser.add_argument(
        "--chunk_overlap", "-v", type=int, default=20, help="Chunk overlap of the embedded texts"
    ) 
    parser.add_argument(
        "--table_k", "-tb", type=int, default=4, help="Number of table nodes to be retrieved" 
    )
    parser.add_argument(
        "--text_k", "-tx", type=int, default=5, help="Number of text nodes to be retrieved" 
    )
    parser.add_argument(
        "--metadata_filter", action='store_true', default=False, help="Whether use metadata filter"
    )
    
    # Response Synthesizer Settings
    parser.add_argument(
        "--synthesizer_llm", "-m", type=str, default="gpt-4", help="LLM used to synthesize responses"
    )
    parser.add_argument(
        "--max_tokens", "-tk", type=int, default=512, help="Max_tokens for LLM generation settings"
    )
    parser.add_argument(
        "--llm_temperature", "-t", type=float, default=0, help="Temperature for LLM generation settings"
    )
    parser.add_argument(
        "--synthesize_prompt_path", "-p", type=str,
        default=os.path.join(MAIN_DIR, "src", "prompts", "llama_index", "rag_v1.py"),
        help="Path to synthesizer unit prompt"
    )
    args = parser.parse_args()
    return args

def main():
    
    args = get_argument_parser()
    
    exp_args = dict(
        emb_type = args.embed_model,
        vectorstore = args.database,
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        table_similarity_top_k = args.table_k,
        text_similarity_top_k = args.text_k,
        index_name = args.index_name,
        description=args.description,
        metadata_filter = args.metadata_filter,
        refine_profile = args.refine_profile,
        refine_llm = args.refine_llm,
        n_iterations = args.n_iterations,
        dataset = args.dataset,
        
        # Generation
        synthesizer_llm = args.synthesizer_llm,
        synthesizer_max_tokens = args.max_tokens,
        synthesizer_temperature = args.llm_temperature,
        response_mode = "simple_summarize",
        synthesize_prompt_path = args.synthesize_prompt_path
    )

    # Setup environment variables
    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]

    save_folder = os.path.join(
        ARTIFACT_DIR, "{}_{}_{}_{}_{}".format(
            exp_args["synthesizer_llm"], exp_args["chunk_size"], exp_args["chunk_overlap"],
            exp_args["description"], datetime.now().strftime("%d-%m-%Y-%H-%M")
        )
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger = get_experiment_logs(exp_args["description"], log_folder=save_folder)
        
    # Prompt
    text_qa_template = CHAT_PROMPT_TEMPLATE

    # Test up test cases
    logger.info("Loading query file from path {}".format(exp_args["dataset"]))
    testcase_df = pd.read_csv(
            exp_args["dataset"],
            usecols = ['ACR scenario', 'Appropriateness Category', 'Scan Order', 'Difficulty', 'Clinical File']
            )

    # patient_profiles = [remove_final_sentence(patient_profile, True)[0].strip() for patient_profile in testcase_df["Clinical File"]]
    # scan_orders = [remove_final_sentence(patient_profile, True)[1].strip() for patient_profile in testcase_df["Clinical File"]]

    extracted_profiles = []
    for testcase in tqdm(testcase_df["Clinical File"], total = len(testcase_df["Clinical File"])):
        extracted_profile = extract_profile_and_scan_order(testcase)
        extracted_profiles.append(extracted_profile)
        
    patient_profiles = [extracted_profile[0].strip() for extracted_profile in extracted_profiles]
    scan_orders = [extracted_profile[1].strip() for extracted_profile in extracted_profiles]
    
    # Setup Query Engine
    db_directory = os.path.join(DATA_DIR, "multimodal-chroma", "descriptions")
    table_index = load_vectorindex(os.path.join(db_directory, "tables"), "chroma")
    text_index = load_vectorindex(os.path.join(db_directory, "texts"), "chroma")
    
    table_retriever = table_index.as_retriever(
        similarity_top_k = exp_args["table_similarity_top_k"]
        )
    text_retriever = text_index.as_retriever(
        similarity_top_k = exp_args["text_similarity_top_k"]
    )
    text_and_table_retriever = CustomCombinedRetriever(
        table_retriever=table_retriever, text_retriever=text_retriever, token_limit = 7000
    )
    
    embs = OpenAIEmbedding()

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(exp_args["synthesizer_llm"]).encode
    )
    callback_manager = CallbackManager([token_counter])

    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            temperature=exp_args["synthesizer_temperature"],
            model=exp_args["synthesizer_llm"], max_tokens=exp_args["synthesizer_max_tokens"]
            ),
        embed_model=embs, callback_manager=callback_manager
    )
    
    response_synthesizer = get_response_synthesizer(
        service_context=service_context, response_mode=exp_args["response_mode"],
        text_qa_template=text_qa_template
    )
    
    query_engine = CustomRetrieverQueryEngine(
        retriever=text_and_table_retriever, response_synthesizer=response_synthesizer,
        callback_manager = CallbackManager([token_counter])
    )

    _, _, _ = run_test_cases(
        testcase_df=testcase_df,
        exp_args=exp_args,
        save_folder=save_folder,
        patient_profiles=patient_profiles,
        scan_orders=scan_orders,
        query_engine=query_engine,
        text_qa_template=text_qa_template,
        logger=logger,
        n_iters=exp_args["n_iterations"]
        )

if __name__ == "__main__":
    main()
    
#python3 src/scripts/exps/combined_db.py -n 3 -desc Combined_retrieval_with_metadata_filter-n=3 -i msk_mri -db data/multimodal-chroma/descriptions --metadata_filter -tb 4 -tx 5 --dataset "data/queries/MSK LLM Fictitious Case Files Full - Reorder.csv"