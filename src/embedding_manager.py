from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from config import config

class EmbeddingManager:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)
        self.vectorstore = None
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=config.embedding_chunk_size,
            chunk_overlap=config.embedding_chunk_overlap,
            length_function=len,
        )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add texts to the vectorstore."""
        documents = self.text_splitter.create_documents(texts, metadatas=metadatas)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        return [doc.page_content for doc in documents]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search."""
        if self.vectorstore is None:
            raise ValueError("No documents have been added to the vectorstore yet.")
        return self.vectorstore.similarity_search(query, k=k)

    def save_vectorstore(self, path: str):
        """Save the vectorstore to disk."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str):
        """Load the vectorstore from disk."""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(path, self.embeddings)
        else:
            loaded_vectorstore = FAISS.load_local(path, self.embeddings)
            self.vectorstore.merge_from(loaded_vectorstore)