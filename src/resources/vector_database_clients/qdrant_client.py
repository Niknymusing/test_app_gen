from langchain_qdrant.qdrant import QdrantVectorStore
from langchain_core.embeddings.embeddings import Embeddings
from typing import Dict, Any, Optional
from langchain_core.vectorstores import VectorStore
import logging
from src.resources.langchain_modules.retriever import Retriever
from src.resources.vector_database_clients.base import BaseVectorDBClient


class QdrantClient(BaseVectorDBClient):
    def __init__(
        self,
        embeddings_model: Embeddings,
        qdrant_url: Optional[str] = None,
        qdrant_port: int = 6333,
        logger: Optional[logging.Logger] = None,
        retriever_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        client_name_key: str = "",  # For logging purposes only
    ):
        super().__init__(embeddings_model=embeddings_model, database_type="qdrant")
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.logger = logger or logging.getLogger(__name__)

        # Store retriever configs keyed by the config keys
        self._retriever_configs: Dict[str, Dict[str, Any]] = retriever_configs or {}

        # Map collection_name to config_key and config for easy lookups
        self._collection_name_to_config = {
            config["collection_name"]: config
            for config in self._retriever_configs.values()
            if "collection_name" in config
        }
        self._collection_name_to_key = {
            config["collection_name"]: key
            for key, config in self._retriever_configs.items()
            if "collection_name" in config
        }
        self._config_key_to_collection_name = {
            key: config["collection_name"]
            for key, config in self._retriever_configs.items()
            if "collection_name" in config
        }

        # Initialize retrievers from existing collections
        client_name = client_name_key or "unknown"
        self.logger.debug(
            f"Attempting connection to Qdrant database with config from client: '{client_name}'..."
        )

        for config in self._retriever_configs.values():
            collection_name = config.get("collection_name")
            try:
                self._initialize_retriever_from_existing_collection(config)
                self.logger.debug(
                    f"Successfully connected to collection: {collection_name}"
                )
            except Exception as e:
                error_msg = f"Failed to initialize retriever for collection '{collection_name}': {e}"
                self.logger.error(error_msg)
                raise

        # Log available collections after initialization
        retrievers_desc = self.list_retrievers(collection_as_main_key=True)
        self.logger.debug(
            f"Available collections for semantic search for client '{client_name}':\n{retrievers_desc}"
        )

    def _initialize_retriever_from_existing_collection(self, config: Dict[str, Any]):
        """Initialize retriever from an existing Qdrant collection."""
        collection_name = config.get("collection_name")
        if not collection_name:
            error_msg = "Each retriever configuration must include a 'collection_name'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(
            f"Attempting to connect to existing collection: {collection_name}"
        )

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings_model,
            collection_name=collection_name,
            url=self.qdrant_url,
            port=self.qdrant_port,
            vector_name=config.get("vector_name", "text"),
            content_payload_key=config.get("content_key", "description"),
            metadata_payload_key=config.get("metadata_key", "metadata"),
        )

        # Register the vector store and build the "Retriever" object for the collection
        self._register_vector_store_and_retriever(collection_name, vector_store, config)

    def _register_vector_store_and_retriever(  # type: ignore
        self, collection_name: str, vector_store: VectorStore, config: Dict[str, Any]
    ):
        self.vector_stores[collection_name] = vector_store
        self.logger.debug(f"Creating retriever for collection '{collection_name}'...")
        retriever = Retriever(
            knowledge_base=vector_store,
            search_kwargs=config.get(
                "search_kwargs", {"score_threshold": 0.5, "k": 10}
            ),
        )
        self.retrievers[collection_name] = retriever
        self.logger.debug(
            f"Retriever for collection '{collection_name}' created successfully."
        )

    def get_retriever(self, config_key: str) -> Retriever:
        collection_name = self._config_key_to_collection_name.get(config_key)
        if not collection_name or collection_name not in self.retrievers:
            raise ValueError(f"Retriever for config key '{config_key}' not found")
        return self.retrievers[collection_name]

    def list_retrievers(  # type: ignore
        self,
        include_search_args: bool = False,
        collection_as_main_key: bool = False,
    ) -> str:
        retriever_strings = []

        for collection_name, retriever in self.retrievers.items():
            config_key = self._collection_name_to_key.get(
                collection_name, collection_name
            )
            config = self._collection_name_to_config.get(
                collection_name, {}
            )  # WARNING: This returns a reference! Avoid mutating directly!

            # Clean unwanted fields
            if not include_search_args:
                config = dict(config)  # Copy so we don't mutate original
                config.pop("search_kwargs", None)

            collection = config.get("collection_name", collection_name)
            description = config.get("description", "").strip().replace("\n", " ")
            vector_name = config.get("vector_name", "text")

            if collection_as_main_key:
                retriever_str = (
                    f"- Collection: {collection}\n"
                    f"  Vector name: {vector_name}\n"
                    f"  Description: {description}"
                )
            else:
                retriever_str = (
                    f"- Name: {config_key}\n"
                    f"  Collection: {collection}\n"
                    f"  Description: {description}"
                )

            retriever_strings.append(retriever_str)

        return "\n\n".join(retriever_strings)
