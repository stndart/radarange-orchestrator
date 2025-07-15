import datetime

from langchain_core.tools import StructuredTool

TOOLNAME = 'time_tool'


def get_time() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


time_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=get_time,
    description='Gets actual current time in local timezone',
)
