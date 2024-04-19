"""Script to run combined db experiment
"""
import argparse
import json
import os
import pandas as pd
import tiktoken

from config import MAIN_DIR, DATA_DIR, ARTIFACT_DIR
from datetime import datetime

from llama_index import ServiceContext, get_response_synthesizer
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

from utils import *
from tqdm import tqdm

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
        "--dataset", type=str, default=os.path.join(DATA_DIR, "queries", "MSK LLM Fictitious Case Files Full - Reorder.csv"), help="path to query dataset"
    )
    parser.add_argument(
        "--json_testcases", "-j", type=str, default=None, help="Path to json files containing patient profiles and scan orders lists"
    )
    # VectorDB & Retrieval settings
    parser.add_argument(
        "--database", "-db", type=str, help="path to database folder",
    )
    parser.add_argument(
        "--index_name", "-i", type=str, help="Name of the index for vector db",
    )
    parser.add_argument(
        "--store_type", type=str, default="simple", help="Type of vector db",
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
        "--text_k", "-tx", type=int, default=3, help="Number of text nodes to be retrieved" 
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
    args = parser.parse_args()
    return args

def main():
    
    args = get_argument_parser()
    
    exp_args = dict(
        store_type = args.store_type,
        db_directory = args.database,
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        similarity_top_k = args.text_k,
        index_name = args.index_name,
        description=args.description,
        n_iterations = args.n_iterations,
        dataset = args.dataset,
        json_testcases = args.json_testcases,
        
        # Generation
        synthesizer_llm = args.synthesizer_llm,
        synthesizer_max_tokens = args.max_tokens,
        synthesizer_temperature = args.llm_temperature,
        response_mode = "compact",
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
    logger.info(
        "Running experiment: {}.\nSaving all artifacts at {}"
        .format(exp_args["description"], save_folder)
    )
    
    # Prompt
    text_qa_template = None
    
    # Setup Query Engine
    ## Setup Retriever
    vector_index = load_vectorindex(
        exp_args["db_directory"],
        emb_store_type=exp_args["store_type"],
        index_name=exp_args["index_name"]
        )
    
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(exp_args["synthesizer_llm"]).encode
    )
    callback_manager = CallbackManager([token_counter])

    embs = OpenAIEmbedding()

    retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=exp_args["similarity_top_k"],
        callback_manager=callback_manager
    )

    ## Setup Synthesizer   
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            temperature=exp_args["synthesizer_temperature"],
            model=exp_args["synthesizer_llm"],
            max_tokens=exp_args["synthesizer_max_tokens"]
            ),
        embed_model=embs, callback_manager=callback_manager
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, response_mode=exp_args["response_mode"],
        text_qa_template=text_qa_template
    )
    
    # Setup QueryEngine
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer,
        callback_manager=callback_manager
    )
    
    # Test up test cases
    logger.info("Loading query file from path {}".format(exp_args["dataset"]))
    testcase_df = pd.read_csv(
            exp_args["dataset"],
            usecols = ['ACR scenario', 'Appropriateness Category', 'Scan Order', 'Difficulty', 'Clinical File']
            )
    
    if exp_args["json_testcases"]:
        logger.info("Loading test cases from {}".format(exp_args["json_testcases"]))
        with open(exp_args["json_testcases"], "r") as f:
            test_case_json = json.load(f)
        patient_profiles = test_case_json["patient_profiles"]
        scan_orders = test_case_json["scan_orders"]

    else:
        logger.info("Cannot find existing test cases. Performing extraction from scratch")
        extracted_profiles = []
        for testcase in tqdm(testcase_df["Clinical File"], total = len(testcase_df["Clinical File"])):
            extracted_profile = extract_profile_and_scan_order(testcase)
            extracted_profiles.append(extracted_profile)
        
        patient_profiles = [extracted_profile[0].strip() for extracted_profile in extracted_profiles]
        scan_orders = [extracted_profile[1].strip() for extracted_profile in extracted_profiles]

        test_case_json = {
            "patient_profiles": patient_profiles, "scan_orders": scan_orders
        }
        with open(os.path.join(DATA_DIR, "queries", "full_testcase_{}.json".format(exp_args["description"])), "w") as f:
            json.dump(test_case_json, f)
        logger.info("Saving test cases to {}".format(os.path.join(DATA_DIR, "queries", "full_testcase_{}.json".format(exp_args["description"]))))    
    
    query_template = (
        "Case: {profile}\n"
        "Scan Ordered: {scan_order}\n"
        "Question: Is this imaging modality for this case USUALLY APPROPRIATE, "
        "MAY BE APPROPRIATE, USUALLY NOT APPROPRIATE or INSUFFICIENT INFORMATION. "
        "Then state precisely the most appropriate imaging modality and if contrast "
        "agent is needed"
    )

    _, _, _ = run_test_cases(
        testcase_df=testcase_df,
        exp_args=exp_args,
        save_folder=save_folder,
        patient_profiles=patient_profiles,
        scan_orders=scan_orders,
        query_engine=query_engine,
        text_qa_template=text_qa_template,
        query_template=query_template,
        logger=logger,
        n_iters=exp_args["n_iterations"]
        )

if __name__ == "__main__":
    main()
    
#python3 src/scripts/exps/basic_db.py -n 5 -desc base_rau_rag-n=5 -i msk_mri -db data/emb_store/simple/openai_512_20 -tx 5 --dataset "data/queries/MSK LLM Fictitious Case Files Full - Reorder.csv" -m gpt-4

#python3 src/scripts/exps/basic_db.py -n 5 -desc base_rau_rag-n=5-k=3 -i msk_mri -db data/emb_store/simple/openai_512_20 -tx 3 --dataset "data/queries/MSK LLM Fictitious Case Files Full.csv" -m gpt-4 -j ./data/queries/full_testcase.json
#python3 src/scripts/exps/basic_db.py -n 5 -desc base_rau_rag-n=5-k=3 -i msk_mri -db data/emb_store/simple/openai_512_20 -tx 3 --dataset "data/queries/MSK LLM Fictitious Case Files Full - Reorder.csv" -m gpt-4 -j ./data/queries/full_testcase_reorder.json