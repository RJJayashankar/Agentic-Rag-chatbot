from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")
    return chunks