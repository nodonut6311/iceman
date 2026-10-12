import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from iceman.tools import (
    load_faculty,
    count_profiles_needed,
    enrich_stored,
    count_images_needed,
    read_stored_email_images,
    save_stored_csv,
    stored_summary,
)

load_dotenv()

MAX_STEPS = 12

SYSTEM_PROMPT = """You are a faculty directory assistant. You build CSV files of faculty members from university faculty pages.

Rules:
- Only report numbers that a tool returned. Never estimate or count yourself.
- Call load_faculty with the URL first. It is cheap, so do it without asking.
- Before enrichment, call count_profiles_needed and tell the user the number. If it is 0, say so and skip the step. If it is more than 0, ask whether to go ahead and wait for a reply. Only call enrich_stored after a yes, using the limit the user approved.
- Before image reads, call count_images_needed and do the same: report the number, skip if 0, otherwise ask and wait. Only call read_stored_email_images after a yes, using the limit the user approved.
- Call save_stored_csv with the path the user gives, or faculty.csv by default.
- Finish by calling stored_summary and reporting its numbers exactly as returned."""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    timeout=120,
    max_retries=0,
)

AGENT_TOOLS = [
    load_faculty,
    count_profiles_needed,
    enrich_stored,
    count_images_needed,
    read_stored_email_images,
    save_stored_csv,
    stored_summary,
]
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