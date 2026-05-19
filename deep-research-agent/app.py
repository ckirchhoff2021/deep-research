import json
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from agent import (
    build_agent_config,
    configure_langsmith,
    create_deep_research_agent,
    get_tracing_context,
)
from dotenv import load_dotenv
import numpy as np

import logging

logger = logging.getLogger("uvicorn.error")

load_dotenv()

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

MAX_ARGS_DISPLAY_LENGTH = 3000
MAX_RESULT_DISPLAY_LENGTH = 3000
MAX_THINKING_DISPLAY_LENGTH = 3000
MAX_FILE_DISPLAY_LENGTH = 36


def get_langsmith_signature():
    """Build a stable cache signature so tracing config changes recreate the agent."""
    config = configure_langsmith()
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or ""
    return {
        "enabled": config["enabled"],
        "project": config["project"],
        "endpoint": endpoint,
        "has_api_key": bool(api_key),
    }


@st.cache_resource
def get_cached_agent(_langsmith_signature: dict):
    logger.info(
        "create agent with langsmith tracing=%s project=%s endpoint=%s has_api_key=%s",
        _langsmith_signature.get("enabled"),
        _langsmith_signature.get("project"),
        _langsmith_signature.get("endpoint"),
        _langsmith_signature.get("has_api_key"),
    )
    return create_deep_research_agent()


def get_session_files():
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def load_session(session_id: str) -> dict:
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"session_id": session_id, "messages": [], "created_at": datetime.now().isoformat()}



def delete_session(session_id: str):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()


def truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def format_session_timestamp(timestamp: str) -> str:
    if not timestamp:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp[:16]

    now = datetime.now()
    if dt.date() == now.date():
        return f"Today · {dt.strftime('%H:%M')}"
    if (now.date() - dt.date()).days == 1:
        return f"Yesterday · {dt.strftime('%H:%M')}"
    return dt.strftime("%Y-%m-%d")


def get_session_preview(messages: list) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = normalize_message_content(msg.get("content", "")).replace("\n", " ").strip()
            if content:
                return truncate_text(content, 72)
    return "No messages yet"


def get_session_records():
    records = []
    for session_file in get_session_files():
        session_id = session_file.stem
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        messages = data.get("messages", [])
        title = data.get("title") or session_id
        created_at = data.get("created_at", "")
        records.append(
            {
                "session_id": session_id,
                "title": truncate_text(title, 42),
                "full_title": title,
                "preview": get_session_preview(messages),
                "message_count": len(messages),
                "created_at": created_at,
                "display_time": format_session_timestamp(created_at),
            }
        )
    return records


def generate_session_title(messages: list) -> str:
    """Generate a title for the session based on the conversation."""
    if not messages:
        return "New Chat"
    
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg.get("content", "")
            break
    
    if not first_user_msg:
        return "New Chat"
    
    if len(first_user_msg) <= 30:
        return first_user_msg
    
    return first_user_msg[:30] + "..."


def save_session(session: dict):
    """Save session with auto-generated title if not present."""
    if "title" not in session or not session["title"]:
        session["title"] = generate_session_title(session.get("messages", []))
    
    session_file = SESSIONS_DIR / f"{session['session_id']}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def format_tool_call(tool_name: str, tool_args: dict, tool_result: str) -> str:
    args_str = str(tool_args)
    if len(args_str) > MAX_ARGS_DISPLAY_LENGTH:
        args_str = args_str[:MAX_ARGS_DISPLAY_LENGTH] + "..."
    result_str = str(tool_result)
    if len(result_str) > MAX_RESULT_DISPLAY_LENGTH:
        result_str = result_str[:MAX_RESULT_DISPLAY_LENGTH] + "..."
    return f"**Tool:** `{tool_name}`\n\n**Args:** `{args_str}`\n\n**Result:**\n```\n{result_str}\n```"


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    if content is None:
        return ""
    return str(content)


def merge_stream_response(existing: str, incoming: str) -> str:
    """Merge streamed text that may arrive as either full snapshots or incremental chunks."""
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming == existing:
        return existing
    if incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing

    max_overlap = min(len(existing), len(incoming))
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(incoming[:overlap]):
            return existing + incoming[overlap:]

    separator = "" if existing.endswith(("\n", " ")) or incoming.startswith(("\n", " ")) else "\n"
    return existing + separator + incoming


def render_process_steps(steps: list, is_complete: bool = True):
    if not steps:
        return

    label = "✅ Trace" if is_complete else "📝 Trace"
    state = "complete" if is_complete else "running"
    with st.status(label, state=state, expanded=not is_complete):
        turn_index = 0
        last_step_index = len(steps) - 1
        for step_index, step in enumerate(steps):
            is_latest_step = not is_complete and step_index == last_step_index
            if step["type"] == "thinking":
                turn_index += 1
                with st.expander(
                    f"💭 Thinking (**:blue[>>> Turn {turn_index}]**)",
                    expanded=is_latest_step,
                ):
                    content = step["content"]
                    if len(content) > MAX_THINKING_DISPLAY_LENGTH:
                        content = content[:MAX_THINKING_DISPLAY_LENGTH] + "..."
                    st.markdown(content)
            elif step["type"] == "tool_call":
                with st.expander(f"🔨 Tool Call (`{step['name']}`)", expanded=is_latest_step):
                    args_str = str(step.get("args", {}))
                    if len(args_str) > MAX_ARGS_DISPLAY_LENGTH:
                        args_str = args_str[:MAX_ARGS_DISPLAY_LENGTH] + "..."
                    result_str = str(step.get("result", ""))
                    if len(result_str) > MAX_RESULT_DISPLAY_LENGTH:
                        result_str = result_str[:MAX_RESULT_DISPLAY_LENGTH] + "..."

                    st.markdown(f"**Tool:** `{step['name']}`")
                    st.markdown(f"**Args:** `{args_str}`")
                    st.markdown(f"**Result:**\n```\n{result_str}\n```")


def display_process_steps(placeholder, steps: list, is_complete: bool = True):
    """Display process steps with each turn wrapped in expander."""
    if not steps:
        return

    with placeholder.container():
        render_process_steps(steps, is_complete=is_complete)


def render_chart(chart_config: dict):
    """Render a chart using Streamlit components."""
    chart_type = chart_config.get("chart_type", "bar")
    data = chart_config.get("data", {})
    title = chart_config.get("title", "")
    x_label = chart_config.get("x_label", "")
    y_label = chart_config.get("y_label", "")
    
    if not data:
        st.warning("No data to display")
        return
    
    df = pd.DataFrame({
        "label": list(data.keys()),
        "value": list(data.values()),
    })
    df = df.set_index("label")
    
    if title:
        st.subheader(title)
    
    if chart_type == "bar":
        st.bar_chart(df, use_container_width=True)
    elif chart_type == "line":
        # {"x1": {"y1": 10, "y2": 20}, "x2": {"y1": 30, "y2": 40}} multi
        # {"x1": [10, 20], "x2": [30, 40]} multi
        # {"x1": 1, "y1": 2, "y2": 3} single
        color=["#FF6B6B", "#45B7D1","#FFCE1B"] * 10
        keys = list(data.keys())
        if isinstance(data[keys[0]], dict):
            x = data[keys[0]].keys()
            xs = {'x' : x}  
            ys = {name : list(data[name].values()) for name in keys}
            xs.update(ys)
            df = pd.DataFrame(xs)
            st.line_chart(df, x='x', color=color[:len(keys)], use_container_width=True)   # color=["#FF6B6B", "#45B7D1","#4ECDC4"], # 红, 蓝, 青
        elif isinstance(data[keys[0]], list):
            x = data[keys[0]]
            xs = {'x' : list(np.arange(len(x)))}  
            xs.update(data)
            df = pd.DataFrame(xs)
            st.line_chart(df, x='x', color=color[:len(keys)], use_container_width=True) 
        else:
            st.line_chart(df, use_container_width=True)
    elif chart_type == "area":
        st.area_chart(df, use_container_width=True)
    elif chart_type == "pie":
        fig, ax = st.pyplot(subplots=True)
        ax.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
        ax.set_title(title)
    elif chart_type == "scatter":
        st.scatter_chart(df, use_container_width=True)
    else:
        st.bar_chart(df, use_container_width=True)
    
    if x_label or y_label:
        st.caption(f"{x_label} vs {y_label}")


def render_table(table_config: dict):
    """Render a table using Streamlit components."""
    headers = table_config.get("headers", [])
    rows = table_config.get("rows", [])
    title = table_config.get("title", "")
    
    if title:
        st.subheader(title)
    
    if headers and rows:
        df = pd.DataFrame(rows, columns=headers)
        st.dataframe(df, hide_index=True)
    else:
        st.warning("No data to display")


def render_file(file_config: dict):
    """Render file as a preview or download card in the chat."""
    file_path = file_config.get("file_path", "")
    display_name = file_config.get("display_name", "download")
    
    if not file_path:
        st.warning("No file path provided")
        return
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        st.error(f"File not found: {file_path}")
        return
    
    if len(display_name) > MAX_FILE_DISPLAY_LENGTH:
        display_name_truncated = display_name[: MAX_FILE_DISPLAY_LENGTH - 3] + "..."
    else:
        display_name_truncated = display_name

    file_extension = file_path_obj.suffix.lower()
    file_icon = {
        ".pdf": "📕",
        ".png": "🖼️",
        ".jpg": "🖼️",
        ".jpeg": "🖼️",
        ".gif": "🖼️",
        ".webp": "🖼️",
        ".bmp": "🖼️",
        ".doc": "📝",
        ".docx": "📝",
        ".xls": "📊",
        ".xlsx": "📊",
        ".csv": "📊",
        ".ppt": "📙",
        ".pptx": "📙",
        ".zip": "🗜️",
        ".rar": "🗜️",
        ".md": "📄",
        ".txt": "📄",
        ".py": "📄",
        ".json": "📄",
        ".mp4": "🎬",
        ".webm": "🎬",
        ".avi": "🎬",
        ".mov": "🎬",
        ".mkv": "🎬",
        ".flv": "🎬",
        ".wmv": "🎬",
    }.get(file_extension, "📎")
    
    # Check if file is an image or video for preview
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    video_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    
    is_image = file_extension in image_extensions
    is_video = file_extension in video_extensions
    
    try:
        with open(file_path_obj, "rb") as f:
            file_data = f.read()

        mime_type, _ = mimetypes.guess_type(str(file_path_obj))
        mime_type = mime_type or "application/octet-stream"
        
        # Render preview if it's image or video
        if is_image:
            st.image(file_data, caption=display_name, use_container_width=True)
        elif is_video:
            st.video(file_data, format=mime_type)
        
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] {
            width: min(360px, 100%) !important;
        }
        div[data-testid="stDownloadButton"] > button {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: min(360px, 100%) !important;
            min-height: 40px !important;
            padding: 7px 12px !important;
            border-radius: 6px !important;
            border: 1px solid var(--border-color, #c7ced8) !important;
            background: transparent !important;
            transition: border-color 0.2s ease, color 0.2s ease !important;
            color: var(--text-color, #2b3440) !important;
            font-weight: 400 !important;
            box-shadow: none !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: transparent !important;
            border-color: var(--text-color, #9aa4b2) !important;
            color: var(--text-color, #111827) !important;
        }
        div[data-testid="stDownloadButton"] > button::after {
            content: "↓";
            color: var(--text-color, #7d8794);
            opacity: 0.7;
            font-size: 12px;
            font-weight: 400;
            margin-left: 10px;
            flex: 0 0 auto;
        }
        div[data-testid="stDownloadButton"] > button p {
            color: var(--text-color, #2b3440) !important;
            font-size: 13px !important;
            text-align: left !important;
            line-height: 1.2 !important;
            margin: 0 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
            max-width: 100% !important;
            flex: 1 1 auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label=f"{file_icon} {display_name_truncated}",
            data=file_data,
            file_name=display_name,
            mime=mime_type,
            key=f"download_{file_path}_{id(file_data)}",
        )
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")


def render_tool_output(result_str: str) -> bool:
    """Parse and render tool output (chart, table, file). Returns True if successful."""
    try:
        config = json.loads(result_str)
        if config.get("type") == "chart":
            render_chart(config)
            return True
        elif config.get("type") == "table":
            render_table(config)
            return True
        elif config.get("type") == "file":
            render_file(config)
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def display_tool_outputs(tool_calls: list, tool_results: list, placeholder=None):
    def _render():
        for i, tc in enumerate(tool_calls):
            result = tool_results[i] if i < len(tool_results) else ""
            tool_name = tc.get("name", "unknown")

            if tool_name in ["prepare_chart_data", "prepare_table_data", "send_file"]:
                render_tool_output(result)

    if placeholder is None:
        _render()
        return

    with placeholder.container():
        _render()


def display_message(msg: dict):
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    tool_results = msg.get("tool_results", [])
    process_steps = msg.get("process_steps", [])

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            if content:
                st.markdown(content)

            display_tool_outputs(tool_calls, tool_results)

            if process_steps:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                render_process_steps(process_steps, is_complete=True)


def main():
    st.set_page_config(
        page_title="Deep Research Agent",
        page_icon="🗄️",
        layout="wide",
    )

    st.markdown("""
    <style>
    .block-container {
        max-width: 940px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stBottomBlockContainer"] {
        max-width: 940px !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stChatInput"] {
        max-width: 940px !important;
        margin: 0 auto !important;
    }

    /* Adjust markdown font sizes in chat messages */
    .stChatMessage h1 { font-size: 1.5rem !important; }
    .stChatMessage h2 { font-size: 1.3rem !important; }
    .stChatMessage h3 { font-size: 1.1rem !important; }
    .stChatMessage h4 { font-size: 1.0rem !important; }
    .stChatMessage p { font-size: 0.95rem !important; }
    .stChatMessage li { font-size: 0.95rem !important; }
    .stChatMessage table { font-size: 0.9rem !important; }
    .stChatMessage code { font-size: 0.85rem !important; }
    
    /* Responsive table styling */
    .stChatMessage .stDataFrame {
        width: fit-content !important;
        max-width: 100%;
    }
    .stChatMessage .stDataFrame > div {
        width: fit-content !important;
        max-width: 100%;
    }
    .stChatMessage [data-testid="stDataFrameResizable"] {
        width: fit-content !important;
        max-width: 100%;
    }
    
    /* Chart styling - balanced width */
    .stChatMessage [data-testid="stVegaLiteChart"] {
        max-width: 600px !important;
    }
    
    /* Sidebar session list buttons - smaller font */
    .session-list .stButton button p {
        font-size: 0.8rem !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] input {
        border-radius: 14px !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border-color: rgba(120, 134, 161, 0.22) !important;
        background: color-mix(in srgb, var(--secondary-background-color) 92%, transparent) !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[kind="tertiary"] {
        border-radius: 12px !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"] {
        border-color: rgba(120, 134, 161, 0.24) !important;
    }

    .session-list [data-testid="stVerticalBlockBorderWrapper"],
    .session-list [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.08rem 0.12rem 0.14rem !important;
        margin-bottom: 0.42rem !important;
        border-radius: 16px !important;
        background: #d9e8ff !important;
        border-color: #4f8fe6 !important;
        box-shadow: 0 8px 22px rgba(59, 130, 246, 0.26), inset 0 0 0 1px rgba(255, 255, 255, 0.38) !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
        overflow: hidden !important;
    }

    .session-list [data-testid="stVerticalBlockBorderWrapper"]:hover,
    .session-list [data-testid="stVerticalBlockBorderWrapper"]:hover > div {
        background: #c6dcff !important;
        border-color: #377ee0 !important;
        box-shadow: 0 12px 26px rgba(59, 130, 246, 0.32), inset 0 0 0 1px rgba(255, 255, 255, 0.42) !important;
    }

    .session-list [data-testid="stVerticalBlockBorderWrapper"]:has(.session-card-badge.current),
    .session-list [data-testid="stVerticalBlockBorderWrapper"]:has(.session-card-badge.current) > div {
        background: #d5f3df !important;
        border-color: rgba(34, 197, 94, 0.56) !important;
        box-shadow: 0 12px 26px rgba(34, 197, 94, 0.26), inset 0 0 0 1px rgba(255, 255, 255, 0.34) !important;
    }

    .session-list button[kind="tertiary"] {
        min-height: 1.95rem !important;
        padding: 0.24rem 0.5rem !important;
        font-size: 0.82rem !important;
        background: transparent !important;
        border: 1px solid rgba(120, 134, 161, 0.18) !important;
    }

    .session-list button[kind="tertiary"] p {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    .sidebar-kicker {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-color);
        opacity: 0.68;
        margin-bottom: 0.15rem;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }

    .sidebar-subtitle {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.72;
        margin-bottom: 0.2rem;
        line-height: 1.45;
    }

    .session-card-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.28rem;
    }

    .session-card-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.74rem;
        border-radius: 999px;
        font-size: 0.98rem;
        font-weight: 600;
        background: color-mix(in srgb, var(--primary-color) 12%, transparent);
        color: var(--primary-color);
    }

    .session-card-badge.current {
        background: color-mix(in srgb, #22c55e 18%, transparent);
        color: #15803d;
    }

    .session-card-time {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.84;
    }

    .session-card-title {
        font-size: 0.97rem;
        font-weight: 400;
        line-height: 1.3;
        margin: 0.06rem 0 0.04rem;
    }

    .session-card-title.current {
        color: #15803d;
        font-weight: 700;
    }

    .session-card-footer {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.84rem;
        color: var(--text-color);
        opacity: 0.78;
        margin-top: 0.18rem;
    }

    .session-active-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #15803d;
        font-weight: 600;
        font-size: 0.84rem;
        opacity: 1;
    }

    .session-footer-meta {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
    }
                 
    .session-footer-separator {
        opacity: 0.42;
    }

    .session-section-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-color);
        opacity: 0.56;
        margin: 0.3rem 0 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

    langsmith_signature = get_langsmith_signature()

    if "session_id" not in st.session_state:
        session_files = get_session_files()
        if session_files:
            latest_session = session_files[0].stem
            st.session_state.session_id = latest_session
        else:
            st.session_state.session_id = str(uuid.uuid4())[:8]

    if "messages" not in st.session_state:
        session = load_session(st.session_state.session_id)
        st.session_state.messages = session.get("messages", [])
        st.session_state.session_title = session.get("title", "")

    if "pending_delete_session_id" not in st.session_state:
        st.session_state.pending_delete_session_id = None

    with st.sidebar:
        session_records = get_session_records()
        st.markdown(
            f"""
            <div class="sidebar-kicker">Workspace</div>
            <div class="sidebar-title">📁 Sessions</div>
            <div class="sidebar-subtitle">{len(session_records)} saved conversations, with safer delete flow.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("➕ New Session", use_container_width=True, key="new_session_btn", type="primary"):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.session_title = ""
            st.session_state.pending_delete_session_id = None
            st.rerun()

        search_query = st.text_input(
            "Search sessions",
            key="session_search",
            placeholder="Search title or first question...",
            label_visibility="collapsed",
        ).strip().lower()

        if search_query:
            filtered_records = [
                record
                for record in session_records
                if search_query in record["full_title"].lower()
                or search_query in record["preview"].lower()
                or search_query in record["display_time"].lower()
            ]
        else:
            filtered_records = session_records

        st.markdown('<div class="session-section-label">Recent Sessions</div>', unsafe_allow_html=True)

        if not filtered_records:
            st.info("No sessions match your search.")

        st.markdown('<div class="session-list">', unsafe_allow_html=True)
        for record in filtered_records[:20]:
            session_id = record["session_id"]
            is_current = session_id == st.session_state.session_id
            with st.container(border=True):
                badge_text = "Current" if is_current else "Saved"
                badge_class = "session-card-badge current" if is_current else "session-card-badge"
                st.markdown(
                    f"""
                    <div class="session-card-meta">
                        <span class="{badge_class}">{badge_text}</span>
                        <span class="session-card-time">{record["display_time"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                button_label = record["title"]
                title_class = "session-card-title current" if is_current else "session-card-title"
                st.markdown(f'<div class="{title_class}">{button_label}</div>', unsafe_allow_html=True)

                footer_cols = st.columns([2.8, 1.15, 1.15])
                with footer_cols[0]:
                    if is_current:
                        footer_html = (
                            f'<div class="session-card-footer"><span class="session-footer-meta">'
                            f'<span class="session-active-indicator">● Active now</span>'
                            f'<span class="session-footer-separator">·</span>'
                            f'<span>{record["message_count"]} messages</span>'
                            f'</span></div>'
                        )
                    else:
                        footer_html = (
                            f'<div class="session-card-footer"><span class="session-footer-meta">'
                            f'<span>{record["message_count"]} messages</span>'
                            f'</span></div>'
                        )
                    st.markdown(footer_html, unsafe_allow_html=True)
                with footer_cols[1]:
                    if not is_current and st.button(
                        "Open",
                        key=f"load_{session_id}",
                        use_container_width=True,
                        type="tertiary",
                    ):
                        st.session_state.session_id = session_id
                        session = load_session(session_id)
                        st.session_state.messages = session.get("messages", [])
                        st.session_state.session_title = session.get("title", "")
                        st.session_state.pending_delete_session_id = None
                        st.rerun()
                with footer_cols[2]:
                    if st.button(
                        "Delete",
                        key=f"prepare_delete_{session_id}",
                        use_container_width=True,
                        type="tertiary",
                    ):
                        st.session_state.pending_delete_session_id = session_id
                        st.rerun()

                if st.session_state.pending_delete_session_id == session_id:
                    st.warning("Delete this session? This action cannot be undone.")
                    confirm_cols = st.columns(2)
                    with confirm_cols[0]:
                        if st.button("Cancel", key=f"cancel_delete_{session_id}", use_container_width=True):
                            st.session_state.pending_delete_session_id = None
                            st.rerun()
                    with confirm_cols[1]:
                        if st.button(
                            "Confirm Delete",
                            key=f"confirm_delete_{session_id}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            delete_session(session_id)
                            st.session_state.pending_delete_session_id = None
                            if is_current:
                                st.session_state.session_id = str(uuid.uuid4())[:8]
                                st.session_state.messages = []
                                st.session_state.session_title = ""
                            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        ### 💡 Tips
        - Ask any question that interests you
        - Examples:
          - "Tell me something about the AuroraEdge-V-2B?"
          - "Who are you ?"
        """)
        st.markdown("""
        ### ⚡ Features
        - **Shell Execute**: Run Python/bash commands
        - **Memory**: Agent remembers context
        - **Skills**: Specialized workflows
        """)

    st.title("💡 Deep Research Agent")
    # st.caption("🌟 Natural language to User-Behavior-Analysis powered by LangChain DeepAgents")

    if langsmith_signature["enabled"]:
        project_name = langsmith_signature["project"] or "default"
        st.caption(
            f"📌 LangSmith Tracing: **:green[on]** | Project: **:green[{project_name}]** | Current Session ID: **:green[{st.session_state.session_id}]**"
        )
    else:
        st.caption("📌 LangSmith Tracing: **:red[off]**")

    for msg in st.session_state.messages:
        display_message(msg)

    if prompt := st.chat_input("Ask a question ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            agent = get_cached_agent(langsmith_signature)
            messages_for_agent = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    messages_for_agent.append({"role": "user", "content": m["content"]})
                elif m["role"] == "assistant":
                    messages_for_agent.append({"role": "assistant", "content": m.get("content", "")})

            response_placeholder = st.empty()
            tool_output_placeholder = st.empty()
            trace_placeholder = st.empty()
            full_response = ""
            tool_calls = []
            tool_results = []
            process_steps = []
            pending_tool_calls = []

            try:
                with st.spinner("🤔 Agent is thinking..."):
                    with get_tracing_context():
                        for event in agent.stream(
                            {"messages": messages_for_agent},
                            config=build_agent_config(st.session_state.session_id),
                            stream_mode="updates",
                        ):
                            has_new_output = False
                            for node_name, node_output in event.items():
                                if isinstance(node_output, dict) and "messages" in node_output:
                                    messages = node_output["messages"]
                                    if isinstance(messages, list):
                                        for msg in messages:
                                            msg_type = type(msg).__name__
                                            if msg_type == "AIMessage":
                                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                                    for tc in msg.tool_calls:
                                                        tool_name = tc.get("name", "unknown")
                                                        tool_call = {
                                                            "name": tool_name,
                                                            "args": tc.get("args", {}),
                                                        }
                                                        tool_calls.append(tool_call)
                                                        pending_tool_calls.append(tool_call)
                                                        has_new_output = True
                                                content_text = normalize_message_content(msg.content)
                                                if content_text:
                                                    display_text = content_text[:MAX_THINKING_DISPLAY_LENGTH] + "..." if len(content_text) > MAX_THINKING_DISPLAY_LENGTH else content_text
                                                    if process_steps and process_steps[-1]["type"] == "thinking":
                                                        process_steps[-1]["content"] = display_text
                                                    elif not process_steps or process_steps[-1] != {
                                                        "type": "thinking",
                                                        "content": display_text,
                                                    }:
                                                        process_steps.append({
                                                            "type": "thinking",
                                                            "content": display_text,
                                                        })
                                                    merged_response = merge_stream_response(full_response, content_text)
                                                    if merged_response != full_response:
                                                        full_response = merged_response
                                                        response_placeholder.markdown(full_response)
                                                        has_new_output = True

                                            elif msg_type == "ToolMessage":
                                                result_text = normalize_message_content(msg.content)
                                                tool_results.append(result_text)
                                                if pending_tool_calls:
                                                    current_tool_call = pending_tool_calls.pop(0)
                                                    display_result = result_text[:MAX_RESULT_DISPLAY_LENGTH] + "..." if len(result_text) > MAX_RESULT_DISPLAY_LENGTH else result_text
                                                    process_steps.append({
                                                        "type": "tool_call",
                                                        "name": current_tool_call["name"],
                                                        "args": current_tool_call["args"],
                                                        "result": display_result,
                                                    })
                                                    has_new_output = True

                            if has_new_output and process_steps:
                                display_process_steps(trace_placeholder, process_steps, is_complete=False)

                if not full_response and tool_results:
                    full_response = tool_results[-1] if tool_results else "Done."

                response_placeholder.markdown(full_response)
                display_tool_outputs(tool_calls, tool_results, placeholder=tool_output_placeholder)
                if process_steps:
                    display_process_steps(trace_placeholder, process_steps, is_complete=True)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                response_placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "process_steps": process_steps,
        })

        session = {
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages,
            "created_at": datetime.now().isoformat(),
        }
        if "session_title" in st.session_state:
            session["title"] = st.session_state.session_title
        save_session(session)


if __name__ == "__main__":
    main()
