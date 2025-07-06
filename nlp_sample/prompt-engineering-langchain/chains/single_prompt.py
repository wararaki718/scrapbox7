from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate


# define template
TEMPLATE = """<s><|user|>
{input_prompt}<|end|>
<|assistant|>"""


def chain_single_prompt(text: str, llm: LlamaCpp) -> str:
    # define a single prompt
    prompt = PromptTemplate(
        input_variables=["input_prompt"],
        template=TEMPLATE,
    )

    # chain
    single_chain = prompt | llm
    response = single_chain.invoke({"input_prompt": text})
    return response
