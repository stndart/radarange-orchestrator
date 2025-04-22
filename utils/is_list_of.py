from typing import Any, Type, TypeVar, TypeGuard, Callable, overload

_T = TypeVar("_T")

def is_list_of(
    seq: list[Any],
    typ: Type[_T]
) -> TypeGuard[list[_T]]:
    """
    Return True if every element of seq is an instance of typ.
    Narrows seq from list[Any] to list[_T] when typ is specified.
    """
    return all(isinstance(item, typ) for item in seq)