from pydantic import BaseModel

from radarange_orchestrator import llm
from radarange_orchestrator.formatting import ResponseFormat

"""
Both llama_cpp and lmstudio backends support structured output for any LLM.
Pass it via formatting.ResponseFormat object, constructable both from pydantic models, as well as json format string.
"""

m = llm()


class Book(BaseModel):
    title: str
    author: str
    year: str


print(m.respond('Tell me about hobbit', response_format=ResponseFormat(Book)).content)
