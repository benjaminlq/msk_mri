from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser

class PatientProfile(BaseModel):
    patient_facts: str
    scan_order: str

extract_template = """You are a radiologist expert. Do not make up additional information.
=========
TASK: You are given a PATIENT PROFILE containing the patient information, medical history, previous diagnosis and the scan order given by clinicians.
You need to extract the following information from the PATIENT PROFILE:
1. Facts related to the patients information, including demographics/previous diagnosis/symptoms
2. Scan order and clinicians suspicions for scan
=========
EXAMPLE:
PATIENT PROFILE: 78 year old Indian Female. Past medical history of diabetes mellitus and hypertension on follow up with polyclinic. Tripped and fell, subsequently unable to walk due to left hip pain. On examination: left limb foreshortening, externally rotated, reduced range of motion due to pain. Nil imaging performed thus far. MRI pelvis and left hip without IV contrast to assess for pelvic/hip fracture and alignment.
ANSWER:
- Patient facts: 78 year old Indian Female. Past medical history of diabetes mellitus and hypertension on follow up with polyclinic. Tripped and fell, subsequently unable to walk due to left hip pain. On examination: left limb foreshortening, externally rotated, reduced range of motion due to pain. Nil imaging performed thus far.
- Scan order: MRI pelvis and left hip without IV contrast to assess for pelvic/hip fracture and alignment.
=========
PATIENT PROFILE: {patient_profile}
"""

def extract_profile_and_scan_order(
    patient_profile: str
):
    llm = OpenAI(model="gpt-4", temperature=0, max_tokens=256)
    extract_program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=PatientProfile),
        prompt_template_str=extract_template, verbose=False,
        llm = llm
    )
    extracted_response = extract_program(patient_profile=patient_profile)
    return (extracted_response.patient_facts, extracted_response.scan_order)