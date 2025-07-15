from langchain_core.messages import InvalidToolCall, ToolCall, ToolMessage
from langchain_core.tools import StructuredTool as Tool

from .download_file_tool import download_tool
from .listdir_tool import ls_tool
from .net_scrape_tool import scrape_tool
from .net_search_tool import net_tool
from .pdf_parse_tool import pdf_tool
from .time_tool import time_tool

# Dynamically collect all Tool instances into a list
all_tools = [value for name, value in globals().items() if isinstance(value, Tool)]

__all__ = [
    'all_tools',
    'time_tool',
    'download_tool',
    'ls_tool',
    'scrape_tool',
    'net_tool',
    'pdf_tool',
    'Tool',
    'ToolMessage',
    'ToolCall',
    'InvalidToolCall',
]
