import json
import os
from typing import Optional, Literal
from langchain_core.tools import tool
from .openai_tools import openai_chat_completion, openai_response


ChartType = Literal["bar", "line", "pie", "scatter", "area"]


@tool
def prepare_chart_data(
    chart_type: str,
    data: dict,
    title: str,
    x_label: str = "",
    y_label: str = "",
) -> str:
    """
    Use this tool when the user asks for a chart, graph, or visualization.
    Creates a beautiful interactive chart rendered in the Streamlit UI.
    
    ALWAYS call this tool after querying data when:
    - User asks for a "chart", "graph", "plot", or "visualization"
    - User says "show me ... visually" or "draw ..."
    - User wants to compare values visually
    
    Args:
        chart_type: Type of chart - must be one of: "bar" (for comparisons), 
                    "line" (for trends), "pie" (for proportions), "scatter" (for correlations), 
                    "area" (for cumulative values)
        data: Dictionary with labels as keys and numeric values as values. 
              Example: {"Iron Maiden": 138.6, "U2": 105.93, "Metallica": 90.09}
        title: Descriptive title for the chart. Example: "Top 5 Artists by Revenue"
        x_label: Label for x-axis (optional). Example: "Artist"
        y_label: Label for y-axis (optional). Example: "Revenue ($)"
    
    Returns:
        JSON string containing chart configuration (will be rendered as interactive chart)
    
    Example usage:
        # plot bar:
        prepare_chart_data(
            chart_type="bar",
            data={"Iron Maiden": 138.6, "U2": 105.93},
            title="Top Artists by Revenue",
            x_label="Artist",
            y_label="Revenue ($)"
        )
        # plot lines:
        prepare_chart_data(
            chart_type="line",
            data={"today": {'1': 100, '2': 200, '3': 300}, "yesterday": {'1': 200, '2': 300, '3': 400}},
            title="Revenue Trend",
            x_label="Time",
            y_label="Revenue ($)"
        )
        prepare_chart_data(
            chart_type="line",
            data={"today": [100, 200, 300], "yesterday": [100, 200, 300]},
            title="Revenue Trend",
            x_label="Time",
            y_label="Revenue ($)"
        )
    """
    chart_config = {
        "type": "chart",
        "chart_type": chart_type,
        "data": data,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }
    return json.dumps(chart_config)


@tool
def prepare_table_data(
    headers: list,
    rows: list,
    title: str = "",
) -> str:
    """
    Prepare data for table display. Returns structured data that will be rendered as a table in the UI.
    
    Args:
        headers: List of column headers
        rows: List of rows, each row is a list of values
        title: Table title (optional)
    
    Returns:
        JSON string containing table configuration
    """
    table_config = {
        "type": "table",
        "headers": headers,
        "rows": rows,
        "title": title,
    }
    return json.dumps(table_config)


@tool
def send_file(
    file_path: str,
    display_name: Optional[str] = None,
    description: str = "",
) -> str:
    """
    Use this tool when you need to send a local file (PDF, PNG, Excel, etc.) to the user.
    The file will be displayed as a downloadable/previewable link in the conversation UI.
    
    Args:
        file_path: Absolute path to the local file. Example: "/data/report.pdf"
        display_name: Custom name to show for the file (optional). If not provided, uses the original filename.
        description: Optional description of what the file contains, shown to the user.
    
    Returns:
        JSON string containing file configuration (will be rendered as a file link in the UI)
    
    Example usage:
        send_file(
            file_path="/data/2024_sales_report.pdf",
            display_name="2024 Sales Report.pdf",
            description="Full annual sales report with breakdown by region"
        )
    """
    # If display name not provided, use the original filename
    if not display_name:
        display_name = os.path.basename(file_path)
    
    file_config = {
        "type": "file",
        "file_path": file_path,
        "display_name": display_name,
        "description": description,
    }
    return json.dumps(file_config)


def get_custom_tools():
    """Return all custom tools."""
    return [
        prepare_chart_data,
        prepare_table_data,
        send_file,
        openai_chat_completion,
        openai_response
    ]

