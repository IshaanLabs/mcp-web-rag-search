from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import websearch
import os
import asyncio
import faiss
from dotenv import load_dotenv

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm:l6-v2")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:latest")


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


async def create_rag(links: list[str]) -> FAISS:
    """Fetch web content from URLs and create a FAISS-GPU vector store."""
    try:
        embeddings = get_embeddings()

        tasks = [websearch.get_web_content(url) for url in links]
        results = await asyncio.gather(*tasks)

        documents = []
        for result in results:
            documents.extend(result)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
        )
        split_documents = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore = move_to_gpu(vectorstore)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag: {str(e)}")
        raise


async def create_rag_from_documents(documents: list[Document]) -> FAISS:
    """Create a FAISS-GPU vector store from pre-fetched documents."""
    try:
        embeddings = get_embeddings()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
        )
        split_documents = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore = move_to_gpu(vectorstore)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag_from_documents: {str(e)}")
        raise


def move_to_gpu(vectorstore: FAISS) -> FAISS:
    """Move FAISS index to GPU if available."""
    try:
        res = faiss.StandardGpuResources()
        cpu_index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        vectorstore.index = gpu_index
        print("FAISS index moved to GPU")
    except Exception as e:
        print(f"GPU not available, using CPU: {e}")
    return vectorstore


async def search_rag(query: str, vectorstore: FAISS) -> list[Document]:
    """Search the vector store for relevant documents."""
    return vectorstore.similarity_search(query, k=3)


async def generate_answer(query: str, vectorstore: FAISS) -> str:
    """Retrieve relevant docs and generate an answer using Ollama LLM."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm
    response = await chain.ainvoke({"context": context, "question": query})
    return response.content





# async def main():
#     # Test search + RAG + generation
#     query = "How many battles were won by maharan amar singh of mewar"

#     print("1. Searching the web...")
#     formatted_results, raw_results = await websearch.search_web(query)
#     print(formatted_results)

#     if not raw_results:
#         print("No results found.")
#         return

#     # Extract URLs
#     urls = [result.get("url") for result in raw_results if result.get("url")]
#     print(f"\n2. Creating RAG from {len(urls)} URLs...")
#     vectorstore = await create_rag(urls)

#     # Test retrieval
#     print("\n3. Retrieving relevant chunks...")
#     docs = await search_rag(query, vectorstore)
#     for i, doc in enumerate(docs, 1):
#         print(f"\n--- Chunk {i} (source: {doc.metadata.get('source', 'unknown')}) ---")
#         print(doc.page_content[:300] + "...")

#     # Test generation
#     print("\n4. Generating answer...")
#     answer = await generate_answer(query, vectorstore)
#     print(f"\nAnswer:\n{answer}")





# if __name__ == "__main__":
#     asyncio.run(main())
