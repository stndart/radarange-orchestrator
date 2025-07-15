import os

from langchain_core.tools import StructuredTool


def ls(path: str = '.') -> list[str]:
    print(f'Called list_directory with path: {path}', flush=True)
    return os.listdir(path)


TOOLNAME = 'list_directory'

ls_tool = StructuredTool.from_function(
    name=TOOLNAME,
    func=ls,
    description='Lists all files and directories in a specified path. \
        Arguments:\
            - path: string - Directory path to list. Defaults to current directory.',
)
