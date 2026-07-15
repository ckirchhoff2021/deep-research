import argparse
import os
import sys
from contextlib import nullcontext
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from tools.custom import get_custom_tools
from rich.console import Console, Group
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.style import Style
import deepagents.middleware.filesystem as fs_middleware


try:
    from langsmith import Client
    from langsmith.run_helpers import tracing_context
except ImportError:  # pragma: no cover - optional dependency at runtime
    Client = None
    tracing_context = None

# Load environment variables
load_dotenv()

console = Console()

fs_middleware.DEFAULT_READ_LIMIT = 3000
READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 3000 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=3000) to see file structure
  - Read more sections: read_file(path, offset=3000, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=3000) reads first 3000 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are returned as multimodal image content blocks (see https://docs.langchain.com/oss/python/langchain/messages#multimodal).

For image tasks:
- Use `read_file(file_path=...)` for `.png/.jpg/.jpeg/.gif/.webp`
- Do NOT use `offset`/`limit` for images (pagination is text-only)
- If image details were compacted from history, call `read_file` again on the same path

- You should ALWAYS make sure a file has been read before editing it."""


def configure_langsmith():
    """Normalize LangSmith/LangChain tracing env vars for compatibility."""
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    if tracing:
        tracing_enabled = str(tracing).lower() in {"1", "true", "yes", "on"}
        if tracing_enabled:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key

    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

    return {
        "enabled": os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true",
        "project": os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT") or "",
        "api_key": os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "",
    }


def create_langsmith_client():
    """Create an explicit LangSmith client when tracing is enabled."""
    config = configure_langsmith()
    if not config["enabled"] or not config["api_key"] or Client is None:
        return None

    client_kwargs = {"api_key": config["api_key"]}
    if config["endpoint"]:
        client_kwargs["api_url"] = config["endpoint"]
    return Client(**client_kwargs)


def get_tracing_context():
    """Return an explicit tracing context or a no-op context."""
    config = configure_langsmith()
    client = create_langsmith_client()
    if not config["enabled"] or client is None or tracing_context is None:
        return nullcontext()
    return tracing_context(
        enabled=True,
        project_name=config["project"] or "default",
        client=client,
    )


def build_agent_config(thread_id: str | None = None) -> dict:
    """Build LangGraph-compatible config so runs can be grouped by thread."""
    resolved_thread_id = thread_id or f"cli-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {"configurable": {"thread_id": resolved_thread_id}}


def create_deep_research_agent():
    """Create a deep research agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Align tracing env vars before creating LangChain objects.
    configure_langsmith()


    # Initialize openai model
    model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                       base_url=os.getenv("API_URL"),
                       api_key=os.getenv("API_KEY"),
                       timeout=300000,
                       streaming=True,
                       temperature=0.3)
    
    # Get custom tools
    custom_tools = get_custom_tools()
    
    # Combine all tools
    all_tools = custom_tools

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,
        memory=["./memory/AGENTS.md"],
        skills=[
            "./skills/"
        ],
        tools=all_tools,
        subagents=[],
        backend=LocalShellBackend(
            root_dir=base_dir,
            virtual_mode=False,
            inherit_env=True,
        ),
    )

    return agent


def format_message_content(content, max_length=500):
    if content is None:
        return ""
    text = str(content)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def display_agent_process(result, console):
    steps = []
    messages = result.get("messages", [])
    
    for i, msg in enumerate(messages[:-1]):
        msg_type = type(msg).__name__
        
        if msg_type == "HumanMessage":
            continue
        elif msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    steps.append({
                        "type": "tool_call",
                        "name": tool_name,
                        "args": tool_args,
                        "content": None
                    })
            elif msg.content:
                steps.append({
                    "type": "thinking",
                    "content": msg.content
                })
        elif msg_type == "ToolMessage":
            if steps and steps[-1]["type"] == "tool_call":
                steps[-1]["result"] = format_message_content(msg.content, 500)
    
    if steps:
        tree = Tree("[bold blue]Agent Process[/bold blue]")
        
        for idx, step in enumerate(steps, 1):
            if step["type"] == "thinking":
                node = tree.add(f"[dim]Step {idx}:[/dim] [italic yellow]Thinking...[/italic yellow]")
                content = format_message_content(step["content"], 500)
                node.add(f"[dim]{content}[/dim]")
            elif step["type"] == "tool_call":
                node = tree.add(f"[dim]Step {idx}:[/dim] [bold cyan]Tool Call: {step['name']}[/bold cyan]")
                args_str = str(step.get("args", {}))
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                node.add(f"[dim]Args: {args_str}[/dim]")
                if step.get("result"):
                    result_node = node.add("[green]Result:[/green]")
                    result_node.add(f"[dim]{step['result']}[/dim]")
        
        console.print(tree)
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="deep-research agents developed by cx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue by country?"
  python agent.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed agent thinking process",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Optional thread id for LangSmith/LangGraph tracing",
    )

    args = parser.parse_args()

    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    tracing_config = configure_langsmith()
    if tracing_config["enabled"]:
        project_name = tracing_config["project"] or "default"
        console.print(f"[dim]LangSmith tracing enabled (project: {project_name}).[/dim]")
    agent = create_deep_research_agent()

    console.print("[dim]Processing query...[/dim]\n")

    try:
        with get_tracing_context():
            result = agent.invoke(
                {"messages": [{"role": "user", "content": args.question}]},
                config=build_agent_config(args.thread_id),
            )

        if args.verbose:
            display_agent_process(result, console)

        final_message = result["messages"][-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
    
