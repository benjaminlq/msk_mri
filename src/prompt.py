from examples import examples
from config import MAIN_DIR
import os
import json
from typing import List

from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.embeddings import OpenAIEmbeddings

instructions = (
"Overall approach to low back pain assessment:\n"
"The assessment of low back pain begins with thorough history taking"
"and physical examination to triage according to the likelihood and type of underlying pathology."
"For low back pain that is likely to originate from the spine (some non-spinal pathologies may present as low back"
"pain, such as abdominal aortic aneurysm, renal disease, or pancreatitis), patients can generally be grouped into"
"those with:\n"
"• Non-specific low back pain, or\n"
"• Low back pain linked to a specific spinal pathology\n"
"A number of investigations, including imaging, are available for further evaluation of low back pain to confirm the"
"diagnosis. The focus of this clinical guidance is on the role of MRI of the lumbar spine for low back pain that is"
"non-specific or linked to a specific spinal pathology.\n"
"------------\n"
"MRI for non-specific low back pain:\n"
"The largest group of patients with low back pain (more than 80%) have non-specific low back pain.4 Typically, a"
"diagnosis of non-specific low back pain is reached when a precise source or defined pathoanatomical cause of"
"pain cannot be identified. For example, patient history and examination do not suggest trauma, nor features of a"
"specific spinal pathology. Sometimes, non-specific low back pain occurs with radicular symptoms in one or both"
"lower limbs, suggesting nerve root involvement.\n"
"------------\n"
"Recommendation 1: Patients with non-specific low back pain, with or without radicular symptoms => MRI is not indicated\n"
"MRI is not recommended for patients with non-specific low back pain (even in the presence of radicular symptoms),"
"particularly at initial presentation. This is because for non-specific low back pain, imaging findings tend to correlate"
"poorly with symptoms, ultimately not altering the management decision or clinical outcomes. For these patients,"
"the initial management approach is usually conservative, and symptoms typically regress soon after onset. Review"
"after 4 to 6 weeks of conservative management (see Recommendation 2 below).\n\n"

"Recommendation 2: Patients with non-specific low back pain despite 4 to 6 weeks of conservative management => MRI may be indicated\n"
"Consider MRI when conservative management of 4 to 6 weeks has not reduced symptoms, as most patients with"
"non-specific low back pain are expected to improve after this.8 Lack of improvement may point to an alternative"
"diagnosis or management approach. When present, certain factors (such as psychological distress, prolonged"
"inactivity, or older age) may perpetuate low back pain and lead to significant disability.9 Therefore, reassess the"
"patient at this point to determine the need for an MRI, including the presence of aforementioned factors and the"
"likelihood of a specific spinal pathology.\n\n"

"Low back pain can be the symptom of a specific spinal pathology, although this happens less commonly than"
"non-specific low back pain. Examples include vertebral fractures, spinal cancer (primary or metastatic),"
"spinal infections, and inflammatory diseases of the spine. The presence of symptoms, signs, previous"
"diagnoses, or other features associated with a specific spinal pathology increases the likelihood of it being"
"the cause (particularly when two or more features are present that suggest the same spinal pathology)\n\n"

"Decision-making regarding further investigations, including imaging, should take into account whether"
"the suspected spinal pathology would require urgent or specialised management (such as surgery), and potential"
"consequences of missing or delaying the diagnosis. When there is significant suspicion of a specific spinal pathology,"
"further investigations are usually indicated.\n\n"

"The choice of imaging modality mainly depends on the type of suspected spinal pathology. MRI is an appropriate"
"choice for investigating soft-tissue pathologies of the spine. The following recommendations pertain to spinal"
"pathologies or clinical features that require MRI of the lumbar spine as diagnostic investigation of low back pain.\n\n"

"Imaging for suspected vertebral fragility fracture:\n"
"Plain radiography is the initial diagnostic investigation of choice for patients with a suspected vertebral"
"fragility fracture, such as elderly patients with a history of low velocity trauma accompanied by osteoporosis"
"or chronic steroid use. Other imaging modalities, including MRI, could be considered after evaluation"
"with plain radiography.\n\n"

"Recommendation 3: Patients with low back pain and progressive neurological symptoms or signs => MRI is indicated\n"
"Progressive neurological deficits, such as deteriorating motor power or worsening numbness, may be due to"
"a space-occupying lesion including herniation of a lumbar intervertebral disc, cancer, infection, and epidural"
"haematoma. MRI is indicated to identify the underlying spinal pathology, especially when the progression is rapid.\n\n"

"Recommendation 4: Patients with low back pain and suspected cauda equina syndrome => MRI is indicated\n"
"Cauda equina syndrome is rare, but warrants urgent management. It may result from disc herniation or other spinal"
"pathologies including cancer, infection, spinal stenosis, spondylolisthesis, and epidural haematoma. Suspect cauda"
"equina syndrome when low back pain presents with associated features, such as bilateral lower limb symptoms"
"or signs (like pain, motor weakness, or sensory changes), sexual dysfunction, bladder or bowel dysfunction, or"
"saddle anaesthesia.\n\n"

"Recommendation 5: Patients with low back pain and cancer or infection (suspected or known) => MRI is indicated\n"
"One should suspect cancer or infection of the spine when a patient with low back pain presents with associated"
"features such as fever or chills, unexplained weight loss, history of cancer, immunosuppression, pain at rest or"
"at night, intravenous drug use, or bacteraemia. MRI can localise the area and extent of disease, and hence it is"
"indicated to investigate lumbar spinal involvement when cancer or infection is suspected.\n\n"

"Recommendation 6: Patients with new or progressive low back pain following an invasive procedure on the lumbar spine => MRI is indicated\n"
"Free disc fragments or scarring may result from invasive spinal procedures. For patients with new or progressive low"
"back pain following an invasive procedure on the lumbar spine, MRI is indicated to examine potential abnormalities,"
"for example to distinguish between scar tissue and recurrent disc herniation.\n\n"
)

system_prompt = """
You are an honest medical clinician. If you do not know an answer, just say "I dont know", do not make up an answer.
======================
TASK
Your role is to refer to the reference information given under CONTEXT, analyse PATIENT PROFILE and recommend if the patient needs an MRI for based on their medication examination.

======================
CONTEXT:
{}

======================
OUTPUT INSTRUCTIONS
Your output should contain an answer and an explanation on whether an MRI should be ordered for the patient.
Answer must be one of [YES, NO]
Explanation must explain why an MRI should be ordered for the patient.
======================
EXAMPLES

======================
""".format(instructions)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "PATIENT PROFILE: {input}"),
        ("ai", "{output}")
    ]
)

def get_prompt(
    embeddings,
    vectorstore: VectorStore = FAISS,
    k: int = 2,
    examples: List = examples   
) -> ChatPromptTemplate:
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples = examples,
        embeddings = embeddings,
        vectorstore_cls = vectorstore,
        k = k    
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=example_prompt,
        example_selector=example_selector,
    )
    
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "PATIENT PROFILE: {input}")
        ]
    )
    
    return final_prompt

if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    
    embs = OpenAIEmbeddings(openai_api_key=api_keys["OPENAI_API_KEY"])
    final_prompt = get_prompt(embs)
    print(final_prompt.format(input = "non-specific patient but already under maintenance for 7 weeks with no improvement"))
    