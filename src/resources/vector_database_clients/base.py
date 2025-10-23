"""Interface Vector Database client class that all child classes should follow"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.embeddings.embeddings import Embeddings
from src.resources.langchain_modules.retriever import Retriever
from langchain_core.vectorstores import VectorStore


class BaseVectorDBClient(ABC):
    def __init__(self, embeddings_model: Embeddings, database_type: str):
        self.embeddings_model = embeddings_model
        self.vector_stores: Dict[str, VectorStore] = {}
        self.retrievers: Dict[str, Retriever] = {}
        self.db_type = database_type

    def get_type(self) -> str:
        """Return type identifier string for the vector DB client."""
        return self.db_type

    @abstractmethod
    def _register_vector_store_and_retriever(
        self, index_name: str, vector_store: VectorStore, config: Dict[str, Any]
    ):
        """Register vector store and create retriever from config.
        NOTE: Should be implemented by subclasses
        """
        pass

    @abstractmethod
    def get_retriever(self, config_key: str) -> Retriever:
        """Return Retriever object for given config key.
        NOTE: Should be implemented by subclasses
        """
        pass

    @abstractmethod
    def list_retrievers(
        self, include_search_args: bool = False, index_as_main_key: bool = False
    ) -> str:
        """Return string description of available retrievers.
        NOTE: Should be implemented by subclasses
        """
        pass
