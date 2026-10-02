from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Literal

from src.tools.web_search import search_tool
from src.state import AgentState
from src.config import settings

class Researcher:
    def __init__(self, api_key: str):
        self.tools = [search_tool]
        self.llm = ChatGroq(
            model=settings.ai,
            api_key=api_key,
            temperature=0.1
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.parser = StrOutputParser()
        
        self.decide_answer_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are an assistant that decides whether to search for information on the Internet. Answer with strictly one word: yes if up-to-date, factual, or fresh information is needed, or no if the question can be answered using general knowledge. No explanations. Only "yes" or "no".'),
            ('human', 'User query: {query}.')
        ])

        self.generate_answer_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a research assistant. Answer the users question based primarily on the information found. IMPORTANT RULES: - Do not invent facts, figures, dates, company valuations, or funding details. - If the information is missing from the search results, state clearly that there is insufficient data. - Do not substitute found data with your own assumptions. - Cite the source for every important factual claim if a URL is available. - Distinguish facts from your own conclusions.'),
            ('human', 'User query: {query}\n, Information found: {search_results}')
        ])

        self.decide_chain = self.decide_answer_prompt | self.llm | self.parser
        self.generate_answer_chain = self.generate_answer_prompt | self.llm | self.parser

        workflow = StateGraph(AgentState)

        workflow.add_node('understand_node', self._node_understand)
        workflow.add_node('decide_node', self._node_decide)
        workflow.add_node('search_node', self._node_search)
        workflow.add_node('generate_answer_node', self._node_generate_answer)

        workflow.add_edge('understand_node', 'decide_node')

        workflow.set_entry_point('understand_node')

        workflow.add_conditional_edges(
            'decide_node',
            self._router,
            {
                'true': 'search_node',
                'false': 'generate_answer_node'
            }
        )

        workflow.add_edge('search_node', 'generate_answer_node')
        workflow.add_edge('generate_answer_node', END)

        self.app = workflow.compile()

    async def _node_understand(self, state: AgentState) -> dict:
        last_message = state['messages'][-1]
        query = last_message.content

        return {'query': query}

    async def _node_decide(self, state: AgentState) -> dict:
        query = state['query']
        result = await self.decide_chain.ainvoke({'query': query})
        answer = result.lower()

        if 'yes' in answer:
            return {'needs_search': True}
        else:
            return {'needs_search': False}

    async def _node_search(self, state: AgentState) -> dict:
        query = state['query']
        result = await search_tool.ainvoke({'query': query})

        return {'search_result': str(result)}

    async def _node_generate_answer(self, state: AgentState) -> dict:
        query = state['query']
        search_results = state.get('search_result', 'No additional data')

        result = await self.generate_answer_chain.ainvoke({'query': query, 'search_results': search_results})

        return {'final_answer': result}

    def _router(self, state: AgentState) -> Literal['true', 'false']:
        result = state['needs_search']

        if result == True:
            return 'true'

        return 'false'