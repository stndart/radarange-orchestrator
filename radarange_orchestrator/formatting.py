from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional, Union

from jsonref import JsonRef
from pydantic import BaseModel

from .config import BACKEND_CAPABILITIES

# Define JSON type for type hints
JsonType = Union[dict[str, Any], list[Any], str, int, float, bool, None]

# Conditional typing for LlamaGrammar
if TYPE_CHECKING:
    from llama_cpp import LlamaGrammar
else:
    LlamaGrammar = Any


def _flatten_refs(obj: JsonRef | dict | list) -> dict:
    """Recursively convert JsonRef/dict/list into plain built-ins"""
    if isinstance(obj, JsonRef):
        obj = dict(obj)  # unwrap the proxy
    if isinstance(obj, dict):
        return {k: _flatten_refs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_flatten_refs(v) for v in obj]
    return obj


def flat_json(model: JsonType | BaseModel) -> JsonType:
    """Flatten JSON schema resolving all $refs"""
    if isinstance(model, type) and issubclass(model, BaseModel):
        model: JsonType = model.model_json_schema()

    # resolve all $refs
    schema_flat: JsonRef = JsonRef.replace_refs(model)
    schema_flat: JsonType = _flatten_refs(schema_flat)
    schema_flat.pop('$defs', dict())

    return schema_flat


class ResponseFormat:
    json_schema: JsonType
    grammar: Optional[LlamaGrammar] = None

    def __init__(self, json_schema: JsonType | BaseModel, keep_reasoning: bool = False):
        self.json_schema = flat_json(json_schema)

        if keep_reasoning and (
            'llama_cpp' not in BACKEND_CAPABILITIES
            or not BACKEND_CAPABILITIES['llama_cpp']['available']
        ):
            raise NotImplementedError(
                'Keep reasoning response format is only for llama_cpp backend'
            )

        if (
            'llama_cpp' in BACKEND_CAPABILITIES
            and BACKEND_CAPABILITIES['llama_cpp']['available']
        ):
            import llama_cpp
            import llama_cpp.llama_grammar

            old_space_rule = llama_cpp.llama_grammar.SPACE_RULE
            llama_cpp.llama_grammar.SPACE_RULE = r'[ \t\n]*'

            grammar_json = llama_cpp.LlamaGrammar.from_json_schema(
                json.dumps(self.json_schema)
            )
            if keep_reasoning:
                base_rules = """
                root ::= "<think>" [^<]+ "</think>" [\\n]* json-schema
                """
            else:
                base_rules = """
                root ::= json-schema
                """
            grammar_json = base_rules + grammar_json._grammar.replace(
                'root', 'json-schema'
            )

            self.grammar = llama_cpp.LlamaGrammar.from_string(grammar_json)
            # restore original space rule
            llama_cpp.llama_grammar.SPACE_RULE = old_space_rule

    def __repr__(self) -> str:
        """Returns visual representation of the response format to be used as prompt hint for an LLM"""
        return json.dumps(self.json_schema, indent=2)
