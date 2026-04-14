from datetime import datetime

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from tools.search import tavily_search
from prompts import RESEARCH_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.7
)

max_concurrent_research_units = 3
max_researcher_iterations = 3

current_date = datetime.now().strftime("%Y-%m-%d")

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent. Give one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search],
}

agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
)


if __name__ == "__main__":
    from langchain.messages import HumanMessage
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What are the main differences between RAG and fine-tuning for LLM applications?"
                )
            ]
        }
    )

    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            print(msg.content)
