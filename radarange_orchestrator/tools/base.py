import inspect
from functools import wraps
from typing import Any, Callable, Optional
from typing_extensions import override

from langchain_core.tools import Tool as LTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun


class Tool(LTool):
    # flag to remember the only argument is placeholder
    zero_args: bool = False

    def __init__(
        self, name: str, func: Optional[Callable], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""

        sig = inspect.signature(func)
        parameters = sig.parameters

        zero_args = False
        if len(parameters) == 0:

            @wraps(func)
            def wrapper(_inp: str):
                return func()

            zero_args = True
        else:
            wrapper = func

        super().__init__(name=name, func=wrapper, description=description, **kwargs)

        self.zero_args = zero_args

    @property
    def args(self) -> dict:
        if getattr(self, 'zero_args', False):
            return dict()
        return super().args

    @override
    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        if getattr(self, 'zero_args', False):
            args = ('',)
        return super()._run(*args, config=config, run_manager=run_manager, **kwargs)

    @override
    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        if getattr(self, 'zero_args', False):
            args = ('',)
        return super()._arun(*args, config=config, run_manager=run_manager, **kwargs)

    @override
    def run(
        self,
        tool_input: str | dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if getattr(self, 'zero_args', False):
            tool_input = ''
        return super().run(tool_input, *args, **kwargs)