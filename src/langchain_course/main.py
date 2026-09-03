from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch


load_dotenv()
# tavily = TavilyClient()

# @tool
# def search_tavily(query: str) -> str: #Tool decorator to convert the function into a tool
#     #Docstring for the tool
#     """
#     Tool that searches the web for information
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching the web for: {query}")
#     #return "This is a test search result"
#     return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
# tools = [search_tavily]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    print(tavily_api_key)
    print("Hello from langchain-course Search Agent Project!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an AI engineer using Langchain in the bay area in the linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()