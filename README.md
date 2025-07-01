# 🧠 Agentic RAG Chatbot

This is an experimental **Agentic AI Chatbot** built using **Retrieval-Augmented Generation (RAG)** and **LangGraph**, powered by the **Mistral API**. It combines document-based search with intelligent agent behavior to provide context-aware answers.

---

## 🚀 Features

- 🔍 **RAG Architecture**: Retrieves relevant chunks from documents to enhance LLM responses.
- 🧠 **Agentic Reasoning**: Uses `LangGraph` to define dynamic, tool-using agents.
- 📄 **Document Ingestion**: Parses, chunks, and embeds documents into a FAISS vector store.
- ⚙️ **Modular Pipeline**: Easy to adapt or extend with new tools, data, or APIs.

---

## 🛠️ Tech Stack

- **Python**
- **LangGraph + LangChain**
- **Mistral API** (via `mistralai` or HTTP)
- **FAISS** (Vector similarity search)
- **Streamlit** *(optional UI)*

---

## 📐 Architecture

```text
User Query
   ↓
LangGraph Agent (Planner + Tools)
   ↓
RAG Pipeline (Retriever + Document Context)
   ↓
Mistral LLM Response


