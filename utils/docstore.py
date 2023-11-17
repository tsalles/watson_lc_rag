import os
import shutil

from typing import Any, List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document


def load_documents(doc_path: str, split_strategy: str = "recursive_char") -> list[Document]:
    docs = PyPDFDirectoryLoader(doc_path).load()
    if split_strategy == "recursive_char":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
    elif split_strategy == "char":
        splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    elif split_strategy == "token":
        splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
    else:
        raise NotImplementedError

    docs = splitter.split_documents(documents=docs)
    return docs


def store_documents(docs: List[Any], embeddings: Embeddings) -> FAISS:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("rag_index")
    return vectorstore.load_local("rag_index", embeddings=embeddings)


def get_vector_store(embeddings: Embeddings, source_docs: str = 'data/', index: str = 'rag_index') -> VectorStore:
    if os.environ["VECSTORE_FRESH_START"] == "true":
        shutil.rmtree("rag_index", ignore_errors=True)
        documents = load_documents("data/")
        return store_documents(documents, embeddings=embeddings)

    try:
        return FAISS.load_local(index, embeddings=embeddings)
    except RuntimeError:
        documents = load_documents(source_docs)
        return store_documents(documents, embeddings=embeddings)

