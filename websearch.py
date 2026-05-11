import asyncio
from dotenv import load_dotenv
import os
from typing import List, Tuple
from langchain_core.documents import Document
from tavily import TavilyClient

load_dotenv(override=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "your_api"))

websearch_config = {
    "parameters": {
        "default_num_results": 5,
    }
}

MAX_RETRIES = 3

async def search_web(query: str, num_results: int = None) -> Tuple[str, list]:
    """Search the web using Tavily and return formatted and raw results."""
    try:
        response = tavily_client.search(
            query=query,
            max_results=num_results or websearch_config["parameters"]["default_num_results"],
            include_answer=True
        )

        results = response.get("results", [])
        formatted = format_search_results(results)
        return formatted, results
    except Exception as e:
        return f"An error occurred while searching: {e}", []

def format_search_results(results):
    if not results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"
    for idx, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")

        markdown_results += f"**{idx}.** [{title}]({url})\n"
        if content:
            markdown_results += f"> **Summary:** {content}\n\n"
        else:
            markdown_results += "\n"

    return markdown_results

async def get_web_content(url: str) -> List[Document]:
    """Get web content using Tavily extract."""
    for attempt in range(MAX_RETRIES):
        try:
            response = tavily_client.extract(urls=[url])

            if response.get("results"):
                return [
                    Document(
                        page_content=r.get("raw_content", ""),
                        metadata={"source": r.get("url", url)}
                    )
                    for r in response["results"]
                ]

            print(f"No content from {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue

        except Exception as e:
            print(f"Error from {url}: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            raise

    return []



# async def main():
#     # Test search
#     query = "History of Maharana Amar Singh"
#     formatted_results, raw_results = await search_web(query)
#     print(formatted_results)
#     print(f"Number of results: {len(raw_results)}")

#     # Test content extraction
#     if raw_results:
#         url = raw_results[0].get("url")
#         print(f"\nExtracting content from: {url}")
#         docs = await get_web_content(url)
#         for doc in docs:
#             print(f"Content length: {len(doc.page_content)} chars")
#             print(f"Preview: {doc.page_content[:200]}...")

# if __name__ == "__main__":
#     asyncio.run(main())



