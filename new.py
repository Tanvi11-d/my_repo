from composio import Composio
from composio_langchain import LangchainProvider
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
 
load_dotenv()
 
composio = Composio(provider=LangchainProvider())
 
llm = ChatGroq(model="openai/gpt-oss-120b")
 
USER_ID = os.getenv("user_id")
 
tools = composio.tools.get(
    user_id=USER_ID,
    tools=["GMAIL_FETCH_EMAILS", "GMAIL_FORWARD_MESSAGE", "GMAIL_GET_DRAFT", "GMAIL_SEND_DRAFT", "GMAIL_SEND_EMAIL"]
)
 
system_prompt = """
When calling any tool:
- NEVER use null
- If field type is:
  - string → use "" (empty string, NEVER null)
  - array → use []
  - boolean → use false
  - integer → use 0
- Always follow the tool schema
- When fetching emails:
  - Limit to max 3 results
  - Return only subject + sender
- Fill required fields properly
- Keep responses small (avoid large outputs)
"""
 
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
 
messages = []
 
while True:
 
    user_input = input("\nYou: ").strip()
 
    if user_input.lower() == "exit":
        break
 
    messages.append({
        "role":"user",
        "content": user_input
    })
 
    print("Assistant: ", end="", flush=True)
 
    try:
 
        result = agent.invoke({
            "messages": messages
        })
 
        assistant_reply = result["messages"][-1].content
 
        print(assistant_reply)
 
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
 
    except Exception as e:
        print(f"\nError: {e}\n")

