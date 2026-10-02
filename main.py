import asyncio
from langchain_core.messages import HumanMessage
from src.graphs.pipeline import Pipeline


async def run():
    pipeline = Pipeline(max_revisions=2)

    result = await pipeline.app.ainvoke({
        'messages': [HumanMessage(content='Какие самые перспективные AI-стартапы в 2026?')]
    })

    print(result.get("final_report"))


if __name__ == "__main__":
    asyncio.run(run())