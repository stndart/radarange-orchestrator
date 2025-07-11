from pydantic import BaseModel

from radarange_orchestrator import llm
from radarange_orchestrator.formatting import ResponseFormat

m = llm()

class Book(BaseModel):
    title: str
    author: str
    year: str

print(m.respond("Tell me about hobbit", response_format=ResponseFormat(Book)).content)
