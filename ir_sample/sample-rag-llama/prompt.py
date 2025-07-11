from langchain_core.prompts import PromptTemplate


# template
TEMPLATE = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""


def get_prompt() -> PromptTemplate:
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=TEMPLATE,
    )
    return prompt
