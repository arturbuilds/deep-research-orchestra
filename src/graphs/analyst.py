from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.config import settings
from src.state import AgentState

class Analyst:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            model=settings.ai,
            api_key=api_key,
            temperature=0.1
        )
        self.parser = StrOutputParser()

        self.analys_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are an analyst. Provide a clear, structured analysis covering: 1. Key facts 2. Main conclusions 3. Risks and uncertainties 4. Areas requiring further verification (if any). Keep it concise, clear, and to the point. No fluff.'),
            ('human', 'User query: {query}\n Information collected by the researcher: {research_data}')
        ])

        self.analys_chain = self.analys_prompt | self.llm | self.parser

        workflow = StateGraph(AgentState)

        workflow.add_node('analysis_node', self._node_analysis)

        workflow.set_entry_point('analysis_node')

        workflow.add_edge('analysis_node', END)

        self.app = workflow.compile()

    async def _node_analysis(self, state: AgentState) -> dict:
        query = state['query']
        research_data = state.get('research_data', 'No data from the researcher')

        result = await self.analys_chain.ainvoke({'query': query, 'research_data': research_data})

        return {'analysis': result}