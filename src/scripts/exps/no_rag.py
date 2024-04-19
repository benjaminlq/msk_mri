"""Script to run combined db experiment
"""
import argparse
import json
import os
import pandas as pd
import tiktoken
import yaml
import re

from copy import deepcopy
from config import MAIN_DIR, DATA_DIR, ARTIFACT_DIR
from prompts.llama_index.no_rag import CHAT_PROMPT_TEMPLATE
from prompts.langchain import CLASSIFICATION_PROMPT

from datetime import datetime

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI as LCChatOpenAI   
from llama_index.llm_predictor import LLMPredictor
from llama_index.callbacks import CallbackManager, TokenCountingHandler
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
        "--dataset", type=str, default=os.path.join(DATA_DIR, "queries", "MSK LLM Fictitious Case Files Full.csv"), help="path to query dataset"
    )
    parser.add_argument(
        "--json_testcases", "-j", type=str, default=None, help="Path to json files containing patient profiles and scan orders lists"
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
        # Experiment
        description=args.description,
        n_iterations = args.n_iterations,
        dataset = args.dataset,
        json_testcases = args.json_testcases,
        
        # Generation
        synthesizer_llm = args.synthesizer_llm,
        synthesizer_max_tokens = args.max_tokens,
        synthesizer_temperature = args.llm_temperature
    )

    # Setup environment variables
    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]

    save_folder = os.path.join(
        ARTIFACT_DIR, "{}_{}_{}".format(
            exp_args["synthesizer_llm"], exp_args["description"], datetime.now().strftime("%d-%m-%Y-%H-%M")
        )
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger = get_experiment_logs(exp_args["description"], log_folder=save_folder)
    logger.info(
        "Running experiment: {}.\nSaving all artifacts at {}"
        .format(exp_args["description"], save_folder)
    )

    logger.info("-------------\nExperiment settings:\n{}".format("\n".join([f"{k}:{v}" for k, v in exp_args.items()])))
    with open(os.path.join(save_folder, "settings.yaml"), "w") as f:
        yaml.dump(exp_args, f)
    logger.info("-------------\nQA PROMPT: {}".format(convert_prompt_to_string(CHAT_PROMPT_TEMPLATE)))
    logger.info("------START RUNNING TEST CASES---------")

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

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(exp_args["synthesizer_llm"]).encode
    )
    callback_manager = CallbackManager([token_counter])

    llm = LLMPredictor(
        llm=OpenAI(
            model=exp_args["synthesizer_llm"],
            temperature=exp_args["synthesizer_temperature"],
            max_tokens=exp_args["synthesizer_max_tokens"]
            ),
        callback_manager=callback_manager
        )
    
    query_template = "Patient Profile: {profile}\nScan ordered: {scan_order}"

    for iter_no in range(exp_args["n_iterations"]):
        
        logger.info(f"Iteration: {iter_no+1}")
        
        responses = []
        queries = []

        for patient_profile, scan_order in tqdm(zip(patient_profiles, scan_orders), total=len(patient_profiles)):
            query_str = query_wrapper(query_template, {"profile": patient_profile, "scan_order": scan_order})
            queries.append(query_str)
            response = llm.predict(CHAT_PROMPT_TEMPLATE, query_str=query_str)
            responses.append(response)

        token_counter = callback_manager.handlers[0]
        logger.info("--------------\nTokens Consumption: Total: {}, Prompt: {}, Completion: {}, Embeddings: {}"
                        .format(token_counter.total_llm_token_count,
                                token_counter.prompt_llm_token_count,
                                token_counter.completion_llm_token_count,
                                token_counter.embedding_token_counts))

        logger.info(f"----------\nTest case Completed. Saving Artifacts into {save_folder}")

        json_responses = []

        fixing_chain = LLMChain(
            llm=LCChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, max_tokens=512),
            prompt=CLASSIFICATION_PROMPT
        )
        
        tk = tqdm(zip(queries, responses, scan_orders), total=len(responses))
        for query, response, scan_order in tk:
            testcase_info = {
                "question": query,
                "result": response,
            }
            answer_query = "Scan Ordered: {}\nAnswer: {}".format(scan_order, testcase_info["result"])
            fixed_answer = fixing_chain(answer_query)

            try:
                appropriateness, recommendation = re.findall(
                #  r"^Appropriateness: ([0-9A-Za-z ]+)\nRecommendation: ([0-9A-Za-z \.]+)$", fixed_answer["text"])[0]
                    r"^[^\n]*Appropriateness: ([^\n]+)\n+[^\n]*Recommendation: ([^\n]+)$", fixed_answer["text"])[0]
            except:
                appropriateness, recommendation = "", ""
                
            testcase_info["appropriateness"] = appropriateness
            testcase_info["recommendation"] = recommendation

            json_responses.append(testcase_info)
            
        with open(os.path.join(save_folder, f"results_{iter_no+1}.json"), "w") as f:
            json.dump(json_responses, f)

        result_df = deepcopy(testcase_df)

        result_df["gpt_raw_answer"] = [response["result"] for response in json_responses]
        result_df["gpt_classification"] = [response["appropriateness"] for response in json_responses]
        result_df["gpt_classification"] = result_df["gpt_classification"].str.upper()
        result_df["gpt_recommendation"] = [response["recommendation"] for response in json_responses]

        result_df = result_df.rename(columns = {"Appropriateness Category": "human_gt"})

        result_df["human_gt"] = result_df["human_gt"].str.replace(r"^UA$", "USUALLY APPROPRIATE", regex=True)
        result_df["human_gt"] = result_df["human_gt"].str.replace(r"^UNA$", "USUALLY NOT APPROPRIATE", regex=True)
        result_df["human_gt"] = result_df["human_gt"].str.replace(r"^MBA$", "MAY BE APPROPRIATE", regex=True)
        result_df["human_gt"] = result_df["human_gt"].str.replace(r"^ICI$", "INSUFFICIENT INFORMATION", regex=True)
        result_df["match"] = (result_df["gpt_classification"] == result_df["human_gt"])

        result_df.to_csv(os.path.join(save_folder, f"result_{iter_no+1}.csv"))
        
        accuracy = result_df["match"].sum() / len(result_df) * 100

        logger.info("------EVALUATION-----")
        logger.info(f"Accuracy score: {accuracy}")
        logger.info(str(result_df.groupby(["gpt_classification", "human_gt"])["match"].value_counts()))
        logger.info(str(result_df.groupby(["human_gt", "gpt_classification"])["match"].value_counts()))

if __name__ == "__main__":
    main()
    
# python3 src/scripts/exps/no_rag.py --description NoRAG -n 2 --dataset data/queries/MSK LLM Fictitious Case Files Full.csv