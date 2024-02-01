from config import GUIDELINES
from langchain.prompts import ChatPromptTemplate as LCChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate

all_guidelines = "\n".join(["- " + guideline for guideline in GUIDELINES])

refine_template = """You are a radiologist expert. Do not make up additional information.
=========
TASK: You are given a PATIENT PROFILE. You need to perform the following information referencing from the PATIENT PROFILE:
1. Extract relevant information for recommendation of imaging procedure, including age, symptomps, previous diagnosis, stage of diagnosis (INITIAL IMAGING OR NEXT STUDY) and suspected conditions, if any.
Only return information given inside the PROFILE, do not make up other information.
2. Return one or more guidelines from the following list of guidelines potentially relevant to the recommendations of imaging procedure given patient profile. If there are no relevant guidelines, return empty list.
{}
=========
OUTPUT INSTRUCTION:
Output your answer as follow:
1. Relevant information:
2. Relevant guidelines: List of guidelines. Match the exact text given in the list. If no relevant guidelines, return [].
=========
""".format(all_guidelines)

human_template = "PATIENT PROFILE: {query_str}"

METADATA_EXTRACT_PROMPT = LCChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(refine_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
)