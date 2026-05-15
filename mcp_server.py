import asyncio
from mcp.server.fastmcp import FastMCP
import rag
import websearch
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="web_search",
    instructions="Web search capability using Tavily API that provides real-time internet search results and uses RAG with Ollama for generating answers."
)


@mcp.tool()
async def search_web_tool(query: str) -> str:
    """Search the web and generate an answer using RAG."""
    logger.info(f"Searching web for query: {query}")
    formatted_results, raw_results = await websearch.search_web(query)

    if not raw_results:
        return "No search results found."

    urls = [result.get("url") for result in raw_results if result.get("url")]
    if not urls:
        return "No valid URLs found in search results."

    vectorstore = await rag.create_rag(urls)
    answer = await rag.generate_answer(query, vectorstore)

    return f"{formatted_results}\n\n### Answer:\n\n{answer}"


@mcp.tool()
async def get_web_content_tool(url: str) -> str:
    """Extract content from a given URL."""
    try:
        documents = await asyncio.wait_for(websearch.get_web_content(url), timeout=15.0)
        if documents:
            return '\n\n'.join(doc.page_content for doc in documents)
        return "Unable to retrieve web content."
    except asyncio.TimeoutError:
        return "Timeout occurred while fetching web content. Please try again later."
    except Exception as e:
        return f"An error occurred while fetching web content: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="sse")
