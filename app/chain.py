from typing import AsyncGenerator

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

from app.config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVER_K,
)


def build_components():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。

コンテキスト:
{context}

質問: {question}
""")
    return retriever, prompt, llm


def build_chain_and_streaming():
    retriever, prompt, llm = build_components()

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        return {
            "context": "\n\n".join(doc.page_content for doc in docs),
            "question": question,
            "sources": list({doc.metadata.get("source", "不明") for doc in docs}),
        }

    chain = RunnableLambda(retrieve_and_format) | RunnableParallel(
        answer=prompt | llm | StrOutputParser(),
        sources=RunnableLambda(lambda x: x["sources"]),
    )

    async def stream_fn(question: str) -> AsyncGenerator[dict, None]:
        docs = retriever.invoke(question)
        sources = list({doc.metadata.get("source", "不明") for doc in docs})
        context = "\n\n".join(doc.page_content for doc in docs)
        async for chunk in (prompt | llm | StrOutputParser()).astream(
            {"context": context, "question": question}
        ):
            yield {"type": "token", "content": chunk}
        yield {"type": "sources", "content": sources}

    return chain, stream_fn
