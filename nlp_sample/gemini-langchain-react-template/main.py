import os

from langchain.agents import AgentExecutor, Tool, create_react_agent, load_tools
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# set environemt variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


SYSTEM_TEMPLATE = """以下の質問に対してできる限り良い回答をしてください。次のツールにアクセスできます：
{tools}
以下の形式を使用してください：
Question: あなたが回答する必要のある質問
Thought: 何をすべきか常に考える
Action: 取るべき行動。以下のいずれかから選択します [{tool_names}]
Action Input: 行動に対する入力
Observation: 行動の結果
...(この思考/行動/行動の入力/観察結果はN回繰り返すことができます)
Thought: 最終回答がわかりました
Final Answer: 元の質問に対する最終回答
開始してください！"""

HUMAN_TEMPLATE = """Question {input}"""

AI_TEMPLATE = """Thought:{agent_scratchpad}"""


def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # prompt
    prompt = ChatPromptTemplate([
        ("ai", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
        ("ai", AI_TEMPLATE)
    ])

    # tool
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="search",
        func=search.run,
        description="Web検索エンジン。一般的な質問に対する検索エンジンとして使用する。",
    )
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(search_tool)

    # agent
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    result: AIMessage = agent_executor.invoke({
        "input": "MacBook Proの現在の価格はUSDでいくらですか? また、為替レートが 1 USD = 150 JPYの場合、JPYではいくらになりますか?",
    })
    print(result)
    print("DONE")


if __name__ == "__main__":
    main()
