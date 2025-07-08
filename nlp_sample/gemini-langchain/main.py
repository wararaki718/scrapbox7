import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt = ChatPromptTemplate([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    result: AIMessage = chain.invoke({
        "input_language": "English",
        "output_language": "Japanese",
        "input": "I love programming.",
    })
    # print(result)
    # print(result.model_dump())
    print(result.content)

    print("DONE")


if __name__ == "__main__":
    main()
