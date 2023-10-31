from langchain.prompts import PromptTemplate as LCPromptTemplate

fix_prompt_template = """TASK: Extract the following information from the provided text query.
1. Appropropriateness of the scan ordered.
2. Most Appropriate Imaging Modality
===============
FORMAT INSTRUCTIONS: Your output should contains the following:
Appropriateness: Can be one of [USUALLY APPROPRIATE, MAY BE APPROPRIATE, USUALLY NOT APPROPRIATE, INSUFFICIENT INFORMATION]
Recommendation: The most appropriate imaging modality. If no appropriate imaging modality, return nothing.
===============
TEXT QUERY: {query}
"""

FIX_PROMPT = LCPromptTemplate.from_template(fix_prompt_template)