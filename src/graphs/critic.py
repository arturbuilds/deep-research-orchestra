from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.config import settings
from src.state import AgentState

class Critic:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model=settings.ai,
            temperature=0.1
        )
        self.parser = StrOutputParser()

        self.critic_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a strict critic of analytical reports. Your task is to evaluate the quality of the final report and decide if it is good enough. Evaluate based on the following points: 1. Completeness — does the report answer the users question? 2. Structure — are there logical headings and clear organization? 3. Factual value — does it contain concrete facts and conclusions rather than fluff? 4. Clarity — is it easy to read? 5. Redundancy — does it contain repetitions, generic phrases, or clutter? Response rules: - If the report is good and there are no significant issues, reply with EXACTLY one line: OK. - If improvements are needed, do NOT write OK. Write specific comments regarding: - what is weak  - what is missing - what should be rewritten or strengthened. Keep it brief and to the point. No introductory remarks.'),
            ('human', 'User query: {query}\n Final report: {report}')
        ])

        self.critic_chain = self.critic_prompt | self.llm | self.parser

        workflow = StateGraph(AgentState)

        workflow.add_node('critic_node', self._node_critic)

        workflow.set_entry_point('critic_node')

        workflow.add_edge('critic_node', END)

        self.app = workflow.compile()

    async def _node_critic(self, state: AgentState) -> dict:
        query = state['query']
        report = state.get('final_report', 'No report')

        is_ok = False
        critique = ''

        answer = await self.critic_chain.ainvoke({'query': query, 'report': report})

        if answer.strip().upper().startswith('OK'):
            return {'is_ok': True, 'critique': 'OK'}

        return {'is_ok': False, 'critique': answer}