from pydantic import BaseModel

from radarange_orchestrator import llm
from radarange_orchestrator.formatting import ResponseFormat

"""
Llama_cpp backend supports complex formats, that allow reasoning as well as structured output.
"""

m = llm(model='*QwQ*Q4*', backend='llama_cpp')


class Book(BaseModel):
    title: str
    author: str
    year: str


print(
    m.respond(
        'Tell me about hobbit',
        response_format=ResponseFormat(Book, keep_reasoning=True),
    ).content
)
