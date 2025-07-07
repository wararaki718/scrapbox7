from langchain.chains.llm import LLMChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


# define template
TEMPLATE = """<s><|user|>Current conversation: {chat_history}

{input_prompt}<|end|>
<|assistant|>"""


def chain_memory_prompt(texts: list[str], llm: LlamaCpp) -> list[dict]:
    # define a single prompt
    prompt = PromptTemplate(
        input_variables=["input_prompt", "chat_history"],
        template=TEMPLATE,
    )

    # model definition
    memory = ConversationBufferMemory(memory_key="chat_history")
    memory_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    # chain
    responses = []
    for text in texts:
        response = memory_chain.invoke({"input_prompt": text})
        responses.append(response)

    return response


def chain_window_memory_prompt(texts: list[str], llm: LlamaCpp) -> list[dict]:
    # define a single prompt
    prompt = PromptTemplate(
        input_variables=["input_prompt", "chat_history"],
        template=TEMPLATE,
    )

    # model definition
    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")
    memory_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    # chain
    responses = []
    for text in texts:
        response = memory_chain.invoke({"input_prompt": text})
        responses.append(response)

    return response
