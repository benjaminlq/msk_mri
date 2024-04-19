from llama_index import PromptTemplate

baseline_query_template = (
    "{query_str}"
    "Question: Is this imaging modality for this case USUALLY APPROPRIATE, "
    "MAY BE APPROPRIATE, USUALLY NOT APPROPRIATE or INSUFFICIENT INFORMATION. "
    "Then state precisely the most appropriate imaging modality and if contrast "
    "agent is needed"
    )

PROMPT_TEMPLATE = PromptTemplate(baseline_query_template)