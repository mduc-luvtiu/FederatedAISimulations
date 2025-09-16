from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

import asyncio
import os

from langchain_ollama.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from mcp.shared.exceptions import McpError

load_dotenv()

model = ChatOllama(model="qwen3:0.6b")  

# Khởi tạo máy chủ Firecrawl
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"]
)

async def main():
    try:
        async with stdio_client(server_params) as (read, write):
            print("Kết nối đến máy chủ Firecrawl đã được thiết lập.")
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Phiên Firecrawl đã được khởi tạo thành công.")
                tools = await load_mcp_tools(session)
                
                agent = create_react_agent(model, tools)

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                    }
                ]

                print("Available Tools - ", *[tool.name for tool in tools])
                print("-" * 60)

                while True:
                    user_input = input("\nYou: ")
                    if user_input == "quit":
                        print("Goodbye")
                        break

                    messages.append({
                        "role": "user",
                        "content": user_input
                    })

                    try:
                        agent_response = await agent.ainvoke({"messages": messages})
                        
                        ai_message = agent_response["messages"][-1].content
                        print("\nAgent: ", ai_message)

                    except Exception as e:
                        print("Error: ", e)
    except McpError as e:
        print(f"Lỗi MCP: {e}")
    except Exception as e:
        print(f"Một lỗi không mong muốn đã xảy ra: {e}")


if __name__ == "__main__":
    asyncio.run(main())
