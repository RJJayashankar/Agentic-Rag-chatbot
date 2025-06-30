from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from mistralai import Mistral
from langgraph.graph import StateGraph
from langchain.schema import Document

from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()  # ✅ Load environment variables from .env

# --- Define LangGraph State ---

class GraphState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    web_results: str
    response: str

# --- Node: Retrieve from Chroma ---

def retriever_node(state: GraphState) -> GraphState:
    query = state["query"]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

# --- Node: Web Search ---

def web_search_node(state: GraphState) -> GraphState:
    query = state["query"]
    search = DuckDuckGoSearchRun()
    web_results = search.run(query)
    return {"web_results": web_results}

# --- Node: Mistral LLM Response ---

def mistral_llm_node(state: dict) -> dict:
    query = state["query"]
    context = ""

    if state.get("retrieved_docs"):
        context += "\n".join([doc.page_content for doc in state["retrieved_docs"][:3]])

    if state.get("web_results"):
        context += f"\nWeb Results:\n{state['web_results']}"

    prompt = (
        f"Answer the following question using the provided context.\n"
        f"Question: {query}\n"
        f"Context:\n{context}\n"
        f"Answer:"
    )

    # ✅ Load API key securely
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("Please set your MISTRAL_API_KEY in your .env file.")

    # ✅ Use Mistral's simplified client (v1.0+)
    client = Mistral(api_key=mistral_api_key)
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response.choices[0].message.content}

# --- LangGraph Workflow ---

graph = StateGraph(GraphState)

graph.add_node("input", lambda state: state)
graph.add_node("retriever", retriever_node)
graph.add_node("web_search", web_search_node)
graph.add_node("mistral_llm", mistral_llm_node)
graph.add_node("output", lambda state: state)

graph.add_edge("input", "retriever")
graph.add_edge("input", "web_search")
graph.add_edge("retriever", "mistral_llm")
graph.add_edge("web_search", "mistral_llm")
graph.add_edge("mistral_llm", "output")

graph.set_entry_point("input")
graph = graph.compile()
