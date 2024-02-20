"""Utility Functions for processing experiment results
"""

from typing import List, Dict, Optional, Union
from llama_index.response.schema import Response
import pandas as pd
import re
import json
from tqdm import tqdm
from copy import deepcopy
import os
from utils.document_utils import convert_doc_to_dict
from prompts.langchain import CLASSIFICATION_PROMPT
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI as LCChatOpenAI
from config import MAIN_DIR

with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
    api_keys = json.load(f)
    
fixing_chain = LLMChain(
    llm=LCChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, max_tokens=512, api_key=api_keys["OPENAI_API_KEY"]),
    prompt=CLASSIFICATION_PROMPT
)

def process_result_json(
    testcase_df: pd.DataFrame, responses: List[Response], save_path: Optional[str] = None
) -> Dict:
    """Convert and save testcases data into json format

    Args:
        testcase_df (pd.DataFrame): Dataframe containing test cases
        responses (List[Response]): List of responses from OpenAI calls
        save_path (Optional[str], optional): Directory of save json. Defaults to None.

    Returns:
        Dict: Dictionary containing results
    """
    json_responses = []
    queries = testcase_df["queries"]
    scan_orders = testcase_df["Scan Order"]
    
    tk = tqdm(zip(queries, responses, scan_orders), total=len(responses))
    for query, response, scan_order in tk:
        testcase_info = {
            "question": query,
            "result": response.response,
            "source_documents": [convert_doc_to_dict(doc) for doc in response.source_nodes]
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
        
    if save_path:
        with open(save_path, "w") as f:
            json.dump(json_responses, f)
    return json_responses

def process_result_df(
    testcase_df: pd.DataFrame, results: Union[List[Dict], List[Response]], save_path: Optional[str] = None
) -> pd.DataFrame:
    """Process result dataframe and match with groundtruth recommendations

    Args:
        testcase_df (pd.DataFrame): Dataframe containing test cases
        results (Union[List[Dict], List[Response]]): List of GPT responses.
        save_path (Optional[str], optional): Directory of save DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: Result dataframe
    """
    if isinstance(results[0], Response):
        results = process_result_json(testcase_df, results)
    
    result_df = deepcopy(testcase_df)
    result_df["gpt_raw_answer"] = [response["result"] for response in results]
    result_df["gpt_classification"] = [response["appropriateness"] for response in results]
    result_df["gpt_classification"] = result_df["gpt_classification"].str.upper()
    result_df["gpt_recommendation"] = [response["recommendation"] for response in results]
    result_df["context"] = [
        "\n\n\n\n".join(["Metadata: {}\nScore: {}\n\nPage Content: {}".format(
            "\n".join([f"{k}: {v}" for k, v in document["metadata"].items()]),
            document["score"],  document["page_content"])
                         for document in response["source_documents"]])
        for response in results
    ]

    result_df = result_df.rename(columns = {"Appropriateness Category": "human_gt"})

    result_df["human_gt"] = result_df["human_gt"].str.replace(r"^UA$", "USUALLY APPROPRIATE", regex=True)
    result_df["human_gt"] = result_df["human_gt"].str.replace(r"^UNA$", "USUALLY NOT APPROPRIATE", regex=True)
    result_df["human_gt"] = result_df["human_gt"].str.replace(r"^MBA$", "MAY BE APPROPRIATE", regex=True)
    result_df["human_gt"] = result_df["human_gt"].str.replace(r"^ICI$", "INSUFFICIENT INFORMATION", regex=True)
    
    result_df["match"] = (result_df["gpt_classification"] == result_df["human_gt"])

    if save_path:
        result_df.to_csv(save_path)

    return result_df