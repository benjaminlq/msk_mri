examples = [
    {
        "input": ("Patient with suspected acute pyelonephritis, first time presentation. "
                  "Uncomplicated patient with no history of pyelonephritis, diabetes, immune compromise, "
                  "no history of stones or renal obstruction, prior renal surgery, advanced age"
                  ),
        "output": ("Answer: NO\n" 
                   "Explanation: Based on guidelines, this patient has non-specific low back pain at initial presentation. "
                   "Based on recommendation 1, MRI is not recommended for this patient."
                   )
        },
    
    {
        "input": "Patient with non-specific low back pain who has been under conservative maintenance for 6 weeks but symptoms do not subside.",
        "output": ("Answer: YES\n"
                   "Explanation: Based on the guildline recommendation 2, it is appropriate to indicate MRI scan for patients "
                   "with non-specific low back pain despite 4-6 weeks of conservative management. as most patients with non-specific "
                   "low back pain are expected to improve after this. Lack of improvement may point to an alternative "
                   "diagnosis or management approach. When present, certain factors (such as psychological distress, prolonged "
                   "inactivity, or older age) may perpetuate low back pain and lead to significant disability"
                   )
        },
    
    {
        "input": "Patient with rapid deteriorating motor power",
        "output": ("Answer: YES\n"
                   "Explanation: The patient has rapid deteriorating motor power, which may indicate a space-occupying lesion. "
                   "Based on recommendation 3, as the patient has signs of progressive neurological deficits, MRI is indicated to "
                   "identify the suspected underlying spinal pathology especially when the progression is rapid."
                   )
        },
    
    {
        "input": "Patient with lower back pain and bilateral lower limb symptoms or signs (like pain, motor weakness, or sensory changes)",
        "output": ("Answer: YES\n"
                   "Explanation: The MRI indication is based on recommendation 4. Based on the symptoms, suspect the patient may have cauda equina"
                   "syndrome, which warrants urgent management and hence MRI is indicated."
                   )
        },
    
    {
        "input": "Patient with low back pain having bowel dysfunction",
        "output": ("Answer: YES\n"
                   "Explanation: The MRI indication is based on recommendation 4. Based on the symptoms, suspect the patient may have cauda equina"
                   "syndrome, which warrants urgent management and hence MRI is indicated."
                   )
        },
    
    {
        "input": "Low back pain patient with fever, unexplained weight loss and history of immunosuppression",
        "output": ("Answer: YES\n"
                   "Explanation: Based on the symptoms, suspect the patient may have cancer or inspection of the spine. "
                   "Based on recommendation 5 MRI can localise the area and extent of disease, and hence it is indicated "
                   "to investigate lumbar spinal involvement when cancer or infection is suspected."
                   )
        },
    
    {
        "input": "Low back pain patient with chills and bacteraemia",
        "output": ("Answer: YES\n"
                   "Explanation: Based on the symptoms, suspect the patient may have cancer or inspection of the spine. "
                   "Based on recommendation 5 MRI can localise the area and extent of disease, and hence it is indicated "
                   "to investigate lumbar spinal involvement when cancer or infection is suspected."
                   )
        },
    
    {
        "input": "Patient with progressive low back pain after an invasive procedure on the lumbar spine",
        "output": ("Answer: YES\n"
                   "Explanation: The MRI indication is based on recommendation 6. The patient may have free disc fragments or scarring "
                   "from invasive spinal procedures."
                   )
        },
]