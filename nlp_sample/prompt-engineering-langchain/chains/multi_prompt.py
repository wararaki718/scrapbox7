from langchain.chains.llm import LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate


# templates
TITLE_TEMPLATE = """<s><|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>"""

CHARACTER_TEMPLATE = """<s><|user|>
Describe the main character of a story about {summary} with the title {title}. Use only two sentences.<|end|>
<|assistant|>"""

STORY_TEMPLATE = """<s><|user|>
Create a story about {summary} with the title {title}. The main character is: {character}.
Only return the story and it cannot be longer than one paragraph.<|end|>
<|assistant|>"""


def chain_multi_prompt(summary: str, llm: LlamaCpp) -> dict[str, str]:
    # define a title prompt and chain
    title_prompt = PromptTemplate(
        input_variables=["summary"],
        template=TITLE_TEMPLATE,
    )
    title_chain = LLMChain(
        llm=llm,
        prompt=title_prompt,
        output_key="title",
    )

    # define a character prompt and chain
    character_prompt = PromptTemplate(
        input_variables=["summary", "title"],
        template=CHARACTER_TEMPLATE,
    )
    character_chain = LLMChain(
        llm=llm,
        prompt=character_prompt,
        output_key="character",
    )

    # define a story prompt and chain
    story_prompt = PromptTemplate(
        template=STORY_TEMPLATE,
        input_variables=["summary", "title", "character"],
    )
    story_chain = LLMChain(
        llm=llm,
        prompt=story_prompt,
        output_key="story",
    )

    # chain
    llm_chain = title_chain | character_chain | story_chain
    response = llm_chain.invoke({"summary": summary})

    return response
