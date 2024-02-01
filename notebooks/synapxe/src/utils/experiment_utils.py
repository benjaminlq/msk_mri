"""Utility Functions for running experiments
"""

import logging
import os
import pandas as pd
import sys
import tiktoken
import yaml

from datetime import datetime
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI as LCChatOpenAI
from langchain.prompts import ChatPromptTemplate as LCChatPromptTemplate

from llama_index import get_response_synthesizer, ServiceContext
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import OpenAI
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import BasePromptTemplate
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

from tqdm import tqdm
from typing import Dict, Optional, List, Sequence, Literal, Tuple

from config import EMB_DIR, ARTIFACT_DIR
from utils.document_utils import load_vectorindex
from utils.prompt_utils import remove_final_sentence, convert_prompt_to_string, query_wrapper
from utils.retrieval_utils import extract_guidelines
from utils.eval_utils import process_result_json, process_result_df

from prompts.langchain import METADATA_EXTRACT_PROMPT
from logging import Logger

def get_experiment_logs(description: str, log_folder: str) -> Logger:
    """Generate logger. By default will log both to file and to terminal.

    Args:
        description (str): Description of the experiment
        log_folder (str): Folder containing the log file

    Returns:
        Logger: logger instance
    """
    logger = logging.getLogger(description)

    stream_handler = logging.StreamHandler(sys.stdout)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    file_handler = logging.FileHandler(filename=os.path.join(log_folder, "logfile.log"))

    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

def setup_query_engine(
    db_directory: str,
    emb_store_type: Literal["simple, faiss"] = "simple",
    index_name: Optional[str] = None,
    similarity_top_k: int = 4,
    text_qa_template: Optional[BasePromptTemplate] = None,
    synthesizer_llm: str = "gpt-3.5-turbo-1106",
    emb_type: str = "openai",
    synthesizer_temperature: int = 0,
    synthesizer_max_tokens: int = 512,
    response_mode: str = "simple_summarize",
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    callback_manager: Optional[CallbackManager] = None,
) ->  BaseQueryEngine:
    """Setup RAG Query enging

    Args:
        db_directory (str): Directory of database(s)
        emb_store_type (Literal[&quot;simple, faiss&quot;], optional): Type of embedding store. Defaults to "simple".
        index_name (Optional[str], optional): Name of index. Defaults to None.
        similarity_top_k (int, optional): Number of documents to be retrieved. Defaults to 4.
        text_qa_template (Optional[BasePromptTemplate], optional): PromptTemplate for RAG. Defaults to None.
        synthesizer_llm (str, optional): Generation LLM in RAG pipeline. Defaults to "gpt-3.5-turbo-1106".
        emb_type (str, optional): Type of embedding model. Defaults to "openai".
        synthesizer_temperature (int, optional): Generation temperature in RAG pipeline. Defaults to 0.
        synthesizer_max_tokens (int, optional): Generation maximum tokens in RAG pipeline. Defaults to 512.
        response_mode (str, optional): Generation response mode in RAG pipeline. Defaults to "simple_summarize".
        node_postprocessors (Optional[List[BaseNodePostprocessor]], optional): Type of postprocessing on retrieved contexts. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): Callback Manager. Defaults to None.

    Returns:
        BaseQueryEngine: RAG Query Engine
    """
    
    vector_index = load_vectorindex(db_directory, emb_store_type=emb_store_type, index_name=index_name)
    
    if emb_type == "openai":
        embs = OpenAIEmbedding()

    retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=similarity_top_k,
        callback_manager=callback_manager
    )

    # Setup Synthesizer
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            temperature=synthesizer_temperature,
            model=synthesizer_llm, max_tokens=synthesizer_max_tokens
            ),
        embed_model=embs, callback_manager=callback_manager
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, response_mode=response_mode,
        text_qa_template=text_qa_template
    )
    
    # Setup QueryEngine
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer,
        node_postprocessors = node_postprocessors
    )
    
    return query_engine

def run_test_cases(
    testcase_df: pd.DataFrame,
    exp_args: Dict,
    testcases: Sequence[str] = None,
    patient_profiles: Sequence[str] = None,
    scan_orders: Sequence[str] = None,
    refined_profiles: Sequence[str] = None,
    relevant_guidelines: Sequence[List[str]] = None,
    query_engine: Optional[BaseQueryEngine] = None,
    query_template: str = "Patient Profile: {profile}\nScan ordered: {scan_order}",
    text_qa_template: Optional[BasePromptTemplate] = None,
    refine_template: Optional[LCChatPromptTemplate] = METADATA_EXTRACT_PROMPT,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    artifact_dir: str = ARTIFACT_DIR,
    emb_folder: str = EMB_DIR,
) -> Tuple:
    """Run experiment test cases

    Args:
        testcase_df (pd.DataFrame): Dataframe containing test cases information
        exp_args (Dict): Experiment arguments
        testcases (Sequence[str], optional): List of test cases. Defaults to None.
        patient_profiles (Sequence[str], optional): List of patient profiles. Defaults to None.
        scan_orders (Sequence[str], optional): List of scan orders. Defaults to None.
        refined_profiles (Sequence[str], optional): List of refined profiles for retrieval query. Use this if you have transformed original queries into another query (e.g HyDE)
            . Defaults to None.
        relevant_guidelines (Sequence[List[str]], optional): List of relevant guidelines metadata (for metadata filtering). Defaults to None.
        query_engine (Optional[BaseQueryEngine], optional): Custom RAG query engine. Defaults to None.
        query_template (str): String template to handle patient profile and scan order. Defaults to "Patient Profile: {profile}\nScan ordered: {scan_order}".
        text_qa_template (Optional[BasePromptTemplate], optional): RAG generation template. Defaults to None.
        refine_template (Optional[LCChatPromptTemplate], optional): This is the template if you want to refine the testcases. Defaults to METADATA_EXTRACT_PROMPT.
        node_postprocessors (Optional[List[BaseNodePostprocessor]], optional): Context nodes postprocessors. Defaults to None.
        artifact_dir (str, optional): Output directory for exp artifacts. Defaults to ARTIFACT_DIR.
        emb_folder (str, optional): Local location of embeddings. Defaults to EMB_DIR.

    Returns:
        Tuple: json_responses, result_df, responses
    """
    save_folder = os.path.join(
        artifact_dir, "{}_{}_{}_{}_{}".format(
            exp_args["synthesizer_llm"],
            exp_args["chunk_size"],
            exp_args["chunk_overlap"],
            exp_args["description"],
            datetime.now().strftime("%d-%m-%Y-%H-%M")
        )
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger = get_experiment_logs(exp_args["description"], log_folder=save_folder)
    
    if not query_engine:
        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(exp_args["synthesizer_llm"]).encode
        )
        callback_manager = CallbackManager([token_counter])
        db_directory = os.path.join(
            emb_folder, exp_args["vectorstore"],
            "{}_{}_{}".format(exp_args["emb_type"], exp_args["chunk_size"], exp_args["chunk_overlap"])
            )
        
        logger.info(f"--------------------\nLoading VectorDB from {db_directory}")
        query_engine = setup_query_engine(
            db_directory,
            emb_store_type=exp_args["vectorstore"],
            index_name=exp_args["index_name"],
            similarity_top_k=exp_args["similarity_top_k"],
            text_qa_template=text_qa_template,
            synthesizer_llm = exp_args["synthesizer_llm"],
            synthesizer_temperature = exp_args["synthesizer_temperature"],
            synthesizer_max_tokens = exp_args["synthesizer_max_tokens"],
            response_mode = "simple_summarize",
            node_postprocessors = node_postprocessors,
            callback_manager = callback_manager
        )
        
    else:
        token_counter = query_engine.callback_manager.handlers[0]

    logger.info(
        "-------------\nExperiment settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in exp_args.items()])
        )
    )

    with open(os.path.join(save_folder, "settings.yaml"), "w") as f:
        yaml.dump(exp_args, f)

    responses = []
    
    logger.info(
        "-------------\nQA PROMPT: {}".format(convert_prompt_to_string(query_engine._response_synthesizer._text_qa_template))
    )

    logger.info(
        "------START RUNNING TEST CASES---------"
    )

    if not (patient_profiles is not None and scan_orders is not None):
        if not testcases:
            testcases = testcase_df["Clinical File"]
        patient_profiles = [remove_final_sentence(testcase, True)[0] for testcase in testcases]
        scan_orders = [remove_final_sentence(testcase, True)[1] for testcase in testcases]

    if exp_args.get("refine_profile") or exp_args.get("metadata_filter"):
        if not (refined_profiles and relevant_guidelines):
            logger.info(
                "-------------\nREFINE PROMPT: {}".format(convert_prompt_to_string(refine_template))
            )
            
            from langchain.callbacks import get_openai_callback
            refine_chain = LLMChain(
                llm=LCChatOpenAI(model_name=exp_args.get("refine_llm", "gpt-3.5-turbo-1106"), temperature=0, max_tokens=512),
                prompt=refine_template)
            
            with get_openai_callback() as cb:
                refined_infos = [extract_guidelines(profile, refine_chain) for profile in tqdm(patient_profiles, total=len(patient_profiles))]
            print(f"Number of refined tokens: Prompt tokens = {cb.prompt_tokens}, Completion tokens = {cb.completion_tokens}")
    
            refined_profiles = [refined_info[0] for refined_info in refined_infos]
            relevant_guidelines = [refined_info[1] for refined_info in refined_infos]
    
    if exp_args.get("refine_profile"):
        patient_profiles = refined_profiles
    
    testcase_df["queries"] = [query_wrapper(query_template, {"profile": patient_profile, "scan_order": scan_order})
                 for patient_profile, scan_order in zip(patient_profiles, scan_orders)]
        
    metadata_filters = relevant_guidelines if exp_args.get("metadata_filter") else [None] * len(testcase_df["queries"])
    
    for query, metadata_filter in tqdm(zip(testcase_df["queries"], metadata_filters), total=len(testcase_df["queries"])):
        input_query = {"str_or_query_bundle": query, "table_filter": metadata_filter, "text_filter": metadata_filter} if metadata_filter is not None else {"str_or_query_bundle": query}
        response = query_engine.query(**input_query)
        responses.append(response)
    
      
    logger.info("--------------\nTokens Consumption: Total: {}, Prompt: {}, Completion: {}, Embeddings: {}"
                .format(token_counter.total_llm_token_count,
                        token_counter.prompt_llm_token_count,
                        token_counter.completion_llm_token_count,
                        token_counter.total_embedding_token_count))

    logger.info(f"----------\nTest case Completed. Saving Artifacts into {save_folder}")
    json_responses = process_result_json(
        testcase_df, responses=responses, save_path=os.path.join(save_folder, "results.json")
        )

    result_df = process_result_df(
        testcase_df, json_responses, save_path=os.path.join(save_folder, "result.csv")
        )

    accuracy = result_df["match"].sum() / len(result_df) * 100

    logger.info("------EVALUATION-----")
    logger.info(f"Accuracy score: {accuracy}")
    logger.info(
        str(result_df.groupby(["gpt_classification", "human_gt"])["match"].value_counts())
    )
    logger.info(
        str(result_df.groupby(["human_gt", "gpt_classification"])["match"].value_counts())
    )

    return json_responses, result_df, responses