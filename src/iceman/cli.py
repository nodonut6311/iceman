from langchain_core.messages import AIMessage, HumanMessage

from iceman.agent import MAX_STEPS, app


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def main():
    print("Faculty directory assistant. Type a request, or 'quit' to exit.\n")
    messages = []
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break

        messages.append(HumanMessage(content=text))
        result = app.invoke({"messages": messages}, config={"recursion_limit": MAX_STEPS})
        messages = result["messages"]

        last = messages[-1]
        if isinstance(last, AIMessage):
            print(f"agent> {_text(last.content)}\n")


if __name__ == "__main__":
    main()