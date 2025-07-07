from langchain.chains.llm import LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory


# template
TEMPLATE="""<s><|user|>Summarize the conversations and update with the new lines.
Current summary:
{summary}
new lines of conversation:
{new_lines}
New summary:<|end|>
<|assistant|>"""


def chain_summary_prompt(texts: list[str], llm: LlamaCpp) -> list[dict]:
    # define a single prompt
    prompt = PromptTemplate(
        input_variables=["new_lines", "summary"],
        template=TEMPLATE,
    )

    # chain
    memory = ConversationSummaryMemory(memory_key="chat_history")
    summary_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    # chain
    responses = []
    for text in texts:
        response = summary_chain.invoke({"input_prompt": text})
        responses.append(response)

    return response
