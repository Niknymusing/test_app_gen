"""Factory interface for setting up embedding models"""

from abc import ABC, abstractmethod
from langchain_core.embeddings.embeddings import Embeddings


class BaseEmbeddingFactory(ABC):
    @abstractmethod
    def initialize_embedding_model(
        self, embedding_cfg: dict, model_secrets: dict = {}
    ) -> Embeddings:
        """
        Abstract method to initialize an embeddings model
        based on the embedding_cfg dictionary and optionally
        model_secrets containing credentials or keys.
        Must be implemented by subclasses.
        """
        pass
