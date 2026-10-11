import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from iceman.tools import (
    load_faculty,
    enrich_stored,
    read_stored_email_images,
    save_stored_csv,
)

load_dotenv()

MAX_STEPS = 12

SYSTEM_PROMPT = """You are a faculty directory assistant. You build CSV files of faculty members from university faculty pages.

Workflow for a request like "make a CSV of faculty from <URL>":
1. Call load_faculty with the URL.
2. Call enrich_stored to fill missing emails and research topics. Use max_profiles of 10 unless the user asks for more.
3. Call read_stored_email_images to read emails from images. Use max_images of 10 unless the user asks for more.
4. Call save_stored_csv with the path the user gives, or faculty.csv by default.

Report the total count and the output file path when done. Never invent people or emails: only use what the tools return."""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    timeout=120,
    max_retries=0,
)

AGENT_TOOLS = [load_faculty, enrich_stored, read_stored_email_images, save_stored_csv]
llm_with_tools = llm.bind_tools(AGENT_TOOLS)


def call_model(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(AGENT_TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")
app = graph.compile()


def run(request: str) -> str:
    """Run the agent on a request. Stops after MAX_STEPS graph steps."""
    result = app.invoke(
        {"messages": [("user", request)]},
        config={"recursion_limit": MAX_STEPS},
    )
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content