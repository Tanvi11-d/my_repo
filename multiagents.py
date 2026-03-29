import os 
from logger import logger 
from dotenv import load_dotenv 
from fastapi import HTTPException 
from composio import Composio 
from composio_langchain import LangchainProvider 
from langchain_groq import ChatGroq 
from langchain.agents import create_agent 
from langchain_community.tools import DuckDuckGoSearchRun,tool 
from langsmith import traceable 

# load env 
load_dotenv() 
GIT_USER_ID = os.getenv("git_user_id") 
USER_ID=os.getenv("user_id") 
api_key=os.getenv("GROQ_API_KEY") 

model = ChatGroq( model="moonshotai/kimi-k2-instruct-0905", api_key=api_key ) 

# web search agent 
search=DuckDuckGoSearchRun() 

@tool 
def web_search(query:str): 
    """Search the web for information.Useful for searching information on the internet.Use this when you need to find current or factual information.""" 
    result=search.invoke(query) 
    return result 

create_web_agent=create_agent( model=model, tools=[web_search] ) 

@tool
def web_agent_tool(query: str):
    """Use for web search queries."""
    res = create_web_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return res["messages"][-1].content

# genearate jokes agent 
@tool 
def generate_joke(topic:str): 
    """Generates a funny joke about a given topic""" 
    return f"tell a joke about given {topic}" 

create_joke_agent=create_agent( 
    model=model,
    tools=[generate_joke],
    system_prompt=f""" 
    - You are a expert at telling funny joke. 
    - follow the below rules. Rules:-
    - Always generate funny jokes. 
    - If a topic is given, make a joke about it. 
    - Be creative with emoji jokes. 
    - Do not give short jokes. 
    - DO not respond only one line jokes. 
    - when user not given topic then you have take own topic jokes. """ ) 


@tool
def joke_agent_tool(query: str):
    """Use for generating jokes."""
    res = create_joke_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return res["messages"][-1].content

# write poem agent
@tool
def write_poem(topic): 
    """write a short poem about given topic. """ 
    return f"write poem about given {topic}" 

create_poem_agent=create_agent( 
    model=model, 
    tools=[write_poem], 
    system_prompt=f""" 
    -You are expert of poem writer.
    - you are follow the below rules. 
    
    Rules:- 
    - always write creative and meaningful poems. 
    - keep clear and well-structured poem. 
    - Write a well-structured poem on the given topic. 
    - make poem simple and unique.return only the poem.
    - Always generate a poem based on the user's query. 
    - Do not return anything except the poem. 
    - Do not include explanations, notes, or text. """)

@tool
def poem_agent_tool(query: str):
    """Use for writing poems."""
    res = create_poem_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return res["messages"][-1].content

composio = Composio( provider=LangchainProvider() ) 

# github agent 
git_tools = composio.tools.get( 
    user_id=GIT_USER_ID, 
    tools=[ "GITHUB_CREATE_A_REPOSITORY_PROJECT", "GITHUB_GET_THE_WEEKLY_COMMIT_COUNT","GITHUB_DELETE_A_FILE" , "GITHUB_UPDATE_A_PROJECT","GITHUB_GET_A_REPOSITORY","GITHUB_GET_REPOSITORY_OWNER", "GITHUB_LIST_COMMITS","GITHUB_RENAME_A_BRANCH","GITHUB_COMMIT_MULTIPLE_FILES" ] ) 

create_git_agent = create_agent( 
    model=model, 
    tools=git_tools, 
    system_prompt=f""" When calling any tool : 
    - If field type is: 
    - string to use " " (empty string, NEVER null). 
    - array -> use []. 
    - boolean -> use True/false. 
    - integer -> use 0. 
    - Always follow the tool schema. 
    - Fill required fields properly. 
    - Keep responses accurate.
    - Perform GitHub actions correctly
    - Always confirm final action clearly
    - Do not return raw JSON """ ) 

@tool
def github_agent_tool(query: str):
    """Use for GitHub operations."""
    res = create_git_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return res["messages"][-1].content

# gmail agent 
gmail_tools = composio.tools.get( 
    user_id=USER_ID,
    tools=[ "GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL", "GMAIL_FORWARD_MESSAGE", "GMAIL_GET_DRAFT", "GMAIL_SEND_DRAFT" ] ) 

create_gmail_agent = create_agent( 
    model=model, 
    tools=gmail_tools, 
    system_prompt = """ When calling any tool: 
    - Keep responses concise and accurate. 
    - If field type is : - string -> use " " (empty string, NEVER null). 
    - array -> use []. - boolean -> use True/False. 
    - integer -> use 0. 
    - Fill required fields properly. 
    - If user say fetch messages then your are show body or subject and from. 
    - Always format professionally
    - When sending an email, always use the provided sender_name in the “Best regards” section; if not provided, use the sender email, and never generate random names.


 """ ) 

@tool
def gmail_agent_tool(query: str):
    """Use for Gmail operations."""
    res = create_gmail_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return res["messages"][-1].content

system_prompt=f""" 
- You are expert multi-agent AI. 
- follow the below rules. 

Rules: 
- Keep answers clear and useful. 
- Do not mention which tool you used. 
- understand user intent and natural language. 
- Decide which sub-agent to call based on user intent. 
- when you are generates jokes with emojis. 

""" 

# subagent = { 
#     "search": create_web_agent, 
#     "jokes": create_joke_agent, 
#     "poem": create_poem_agent, 
#     "git": create_git_agent, 
#     "gmail": create_gmail_agent } 

# @traceable(name="task") 
# @tool
# def task(query): 
#     """ Smart dispatcher that routes user query to correct sub-agent. 
#     Handles: 
#     - poem writing 
#     - joke generation 
#     - gmail actions (send, fetch, get etc.) 
#     - github actions
#     - web search """  

#     agents=subagent[query] 
#     result=agents.invoke({"messages": [{"role": "user", "content": query}]}) 
#     return result["messages"][-1].content 

# main_agent=create_agent( 
#     model=model, 
#     tools=[task], 
#     system_prompt=system_prompt ) 

main_agent = create_agent(
    model=model,
    tools=[
        web_agent_tool,
        joke_agent_tool,
        poem_agent_tool,
        github_agent_tool,
        gmail_agent_tool
    ],

    system_prompt="""
You are an intelligent multi-agent AI system.

You have:
- joke agent
- poem agent
- gmail agent
- github agent
- web search agent

IMPORTANT RULES:

1. Understand user intent clearly.

2. If task requires multiple steps:
   - First call required agent
   - Then use output in next agent

Example:
- "send joke to email"
  Step 1: generate joke
  Step 2: send email with joke

3. Always return FINAL CLEAN RESPONSE.

4. Never return raw tool output.

5. Response rules:
- Joke → return joke
- Poem → return poem
- Email → confirm sent
- GitHub → confirm action

6. Do NOT mention tools.

7. Be smart and chain agents when required.
"""
)
     
@traceable(name="call_agent")
def call_agent(query: str):
    try:
        logger.info("Running main agent...")

        response = main_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        logger.info("Response generated")

        return response["messages"][-1].content

    except Exception as e:
        logger.error(f"Agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))