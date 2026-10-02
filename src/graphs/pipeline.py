from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.graphs.researcher import Researcher
from src.graphs.analyst import Analyst
from src.graphs.writer import Writer
from src.graphs.critic import Critic
from src.state import AgentState
from src.config import settings


class Pipeline:
    def __init__(self, max_revisions: int = 2):
        self.max_revisions = max_revisions

        self.researcher = Researcher(api_key=settings.groq_key)
        self.analyst = Analyst(api_key=settings.groq_key)
        self.writer = Writer(api_key=settings.groq_key)
        self.critic = Critic(api_key=settings.groq_key)

        workflow = StateGraph(AgentState)

        workflow.add_node('research_node', self._node_research)
        workflow.add_node('analyst_node', self._node_analyst)
        workflow.add_node('writer_node', self._node_writer)
        workflow.add_node('critic_node', self._node_critic)

        workflow.set_entry_point('research_node')

        workflow.add_edge('research_node', 'analyst_node')
        workflow.add_edge('analyst_node', 'writer_node')
        workflow.add_edge('writer_node', 'critic_node')

        workflow.add_conditional_edges(
            'critic_node',
            self._critic_router,
            {
                'OK': END,
                'NOT': 'writer_node',
            },
        )

        self.app = workflow.compile()

    async def _node_research(self, state: AgentState) -> dict:
        print('1. Researcher...')

        if state.get('query'):
            query = state['query']
        else:
            query = state['messages'][-1].content

        research_result = await self.researcher.app.ainvoke({'messages': [HumanMessage(content=query)]})

        research_data = research_result.get('final_answer')

        return {
            'query': research_result.get('query', query),
            'research_data': research_data,
            'revision_count': 0,
        }

    async def _node_analyst(self, state: AgentState) -> dict:
        print('2. Analyst...')

        analyst_result = await self.analyst.app.ainvoke({
            'query': state['query'],
            'research_data': state.get('research_data', 'No research data'),
        })

        return {
            'analysis': analyst_result['analysis']
        }

    async def _node_writer(self, state: AgentState) -> dict:
        revision = state.get('revision_count', 0)
        print(f'3. Writer (attempt {revision + 1})...')

        writer_input = {
            'query': state['query'],
            'analysis': state.get('analysis', 'No analysis'),
        }

        writer_result = await self.writer.app.ainvoke(writer_input)

        return {
            'final_report': writer_result['final_report']
        }

    async def _node_critic(self, state: AgentState) -> dict:
        print('4. Critic...')

        critic_result = await self.critic.app.ainvoke({
            'query': state['query'],
            'final_report': state.get('final_report', 'No report'),
        })

        is_ok = critic_result.get('is_ok', False)
        critique = critic_result.get('critique', '')

        revision_count = state.get('revision_count', 0)
        if not is_ok:
            revision_count += 1

        print('Critic:', 'OK' if is_ok else 'NOT OK')
        if not is_ok:
            print(critique)

        return {
            'is_ok': is_ok,
            'critique': critique,
            'revision_count': revision_count,
        }

    def _critic_router(self, state: AgentState) -> Literal['OK', 'NOT']:
        is_ok = state.get('is_ok', False)
        revision_count = state.get('revision_count', 0)

        if is_ok:
            return 'OK'

        if revision_count >= self.max_revisions:
            print('Max revisions reached. Stopping.')
            return 'OK'

        return 'NOT'