import asyncio
from compare_pipeline import run

async def main():
    await asyncio.gather(
        run("data/doc001.pdf"),
        run("data/doc002.pdf"),
        run("data/doc003.pdf"),
        run("data/ai_queue_design.pdf")
    )

asyncio.run(main())
