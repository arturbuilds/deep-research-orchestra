from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.config import settings
from src.state import AgentState

class Writer:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model=settings.ai,
            temperature=0.2
        )
        self.parser = StrOutputParser()

        self.writer_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a professional technical writer. Write a clean, well-structured final report. - Use headings - Make the text readable - Avoid unnecessary fluff - Maintain important facts and conclusions - Style: professional yet clear'),
            ('human', 'User query: {query}\n Analysis: {analysis}')
        ])

        self.writer_chain = self.writer_prompt | self.llm | self.parser

        workflow = StateGraph(AgentState)

        workflow.add_node('writer_node', self._node_writer)

        workflow.set_entry_point('writer_node')

        workflow.add_edge('writer_node', END)

        self.app = workflow.compile()

    async def _node_writer(self, state: AgentState) -> dict:
        query = state['query']
        analysis = state.get('analysis', 'No analysis')

        answer = await self.writer_chain.ainvoke({'query': query, 'analysis': analysis})

        return {'final_report': answer}