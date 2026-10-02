from typing import TypedDict
from langgraph.graph import add_messages
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    needs_search: bool
    search_result: str
    final_answer: str
    research_data: str
    analysis: str
    final_report: str
    is_ok: bool
    critique: str
    revision_count: int