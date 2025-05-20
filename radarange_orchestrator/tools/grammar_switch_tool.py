from typing import Optional
from llama_cpp import LlamaGrammar
from ..types.tools import FunctionDescription, Tool, ToolResult, ToolDef

grammar_tool_def = ToolDef(
    type = "function",
    function = FunctionDescription(
        name = "switch_grammar_to_output",
        description = "Switches grammar to the output format for the next generation",
    )
)

class GrammarContext:
    def __init__(self, name: str, current_grammar: Optional[LlamaGrammar]):
        self.grammar: Optional[LlamaGrammar] = current_grammar
        self.name: str = name
    
    def switch_grammar(self, new_grammar: Optional[LlamaGrammar]):
        print("Called switch grammar", flush=True)
        self.grammar = new_grammar
    
    def clear_grammar(self):
        self.grammar = None
    
    def get_current_grammar(self) -> Optional[LlamaGrammar]:
        return self.grammar

def create_set_grammar_tool(grammar: Optional[LlamaGrammar] = None) -> tuple[GrammarContext, Tool]:
    grammar_context = GrammarContext(grammar_tool_def.function.name, None)

    def handler() -> ToolResult:
        grammar_context.switch_grammar(grammar)
        return ToolResult(
            status = "success",
            stdout = "Switched grammar to output format for the next generation"
        )

    return grammar_context, Tool(
        definition = grammar_tool_def,
        handler = handler
    )