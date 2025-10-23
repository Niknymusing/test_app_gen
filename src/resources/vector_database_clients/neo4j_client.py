from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings.embeddings import Embeddings
from typing import Dict, Any, Optional
from langchain_core.vectorstores import VectorStore
from abc import ABC
import logging
from src.resources.langchain_modules.retriever import Retriever
from src.resources.vector_database_clients.base import BaseVectorDBClient


class Neo4jClient(BaseVectorDBClient):
    def __init__(
        self,
        embeddings_model: Embeddings,
        neo4j_url: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
        logger: Optional[logging.Logger] = None,
        retriever_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        client_name_key: str = "",  # For logging purposes only
    ):
        super().__init__(embeddings_model=embeddings_model, database_type="neo4j")
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.logger = logger or logging.getLogger(__name__)

        # Connect to Neo4j and refresh schema
        client_name = client_name_key or "unknown"
        self.logger.debug(
            f"Attempting connection to the Neo4j database with config from client: '{client_name}'..."
        )
        self.graph_db = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.neo4j_database,
            enhanced_schema=True,
            refresh_schema=True,
        )
        self.logger.debug("Retrieving and storing graph schema...")
        self.schema_details = self.graph_db.schema

        # Store retriever configs keyed by the config keys ("voting_proposal_store", etc)
        self._retriever_configs: Dict[str, Dict[str, Any]] = retriever_configs or {}

        # Map index_name to config_key and config for easy lookups
        self._index_name_to_config = {
            config["index_name"]: config
            for config in self._retriever_configs.values()
            if "index_name" in config
        }
        self._index_name_to_key = {
            config["index_name"]: key
            for key, config in self._retriever_configs.items()
            if "index_name" in config
        }
        self._config_key_to_index_name = {
            key: config["index_name"]
            for key, config in self._retriever_configs.items()
            if "index_name" in config
        }

        # Always try to connect to existing indexes first, fallback to graph-based if fail
        for config in self._retriever_configs.values():
            index_name = config.get("index_name")
            try:
                self._initialize_retriever_from_existing_index(config)
            except Exception:
                self.logger.debug(
                    f"Existing index not found: {index_name}, creating index from the existing graph..."
                )
                self._initialize_retriever_from_existing_graph(config)

        # Log available indices after initilization
        retrievers_desc = self.list_retrievers(index_as_main_key=True)
        self.logger.debug(
            f"Available indices for semantic search for client '{client_name}':\n{retrievers_desc}"
        )

    def get_graph_schema_details(self) -> str:
        """Returns a detailed string with the detailed graph schema"""
        return self.schema_details

    def _initialize_retriever_from_existing_index(self, config: Dict[str, Any]):
        """Attempt to connect to an existing Neo4j vector index."""
        index_name = config.get("index_name")
        if not index_name and isinstance(index_name, str):
            raise TypeError(
                "An 'index_name' that is not a string was passed to the '_initialize_retriever_from_existing_index' function."
            )
        self.logger.debug(
            f"Attempting to connect to an existing index with name: {index_name}"
        )

        vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings_model,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.neo4j_database,
            index_name=index_name,  # type: ignore
            retrieval_query=config.get("retrieval_query"),
        )
        # Found an index...
        self.logger.debug(f"Found an existing index with name: {index_name}")

        # Register the vector store and build the "Retriever" object for the index
        self._register_vector_store_and_retriever(index_name, vector_store, config)  # type: ignore

    def _initialize_retriever_from_existing_graph(self, config: Dict[str, Any]):
        index_name = config.get("index_name")
        if not index_name:
            error_msg = "Each retriever configuration must include an 'index_name'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.logger.debug(
                f"Creating vector store for index '{index_name}' from existing graph..."
            )
            vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings_model,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                database=self.neo4j_database,
                index_name=index_name,
                node_label=config.get("node_label", ""),
                text_node_properties=config.get("text_node_properties", []),
                embedding_node_property=config.get("embedding_node_property"),  # type: ignore
                retrieval_query=config.get("retrieval_query"),  # type: ignore
            )

            # Register the vector store and build the "Retriever" object for the index
            self._register_vector_store_and_retriever(index_name, vector_store, config)

        except Exception as e:
            error_msg = f"Failed to initialize retriever '{index_name}': {e}"
            self.logger.error(error_msg)
            raise

    def _register_vector_store_and_retriever(
        self, index_name: str, vector_store: VectorStore, config: Dict[str, Any]
    ):
        self.vector_stores[index_name] = vector_store
        self.logger.debug(f"Creating retriever for index '{index_name}'...")
        retriever = Retriever(
            knowledge_base=vector_store,
            search_kwargs=config.get(
                "search_kwargs", {"score_threshold": 0.5, "k": 10}
            ),
        )
        self.retrievers[index_name] = retriever
        self.logger.debug(f"Retriever for index '{index_name}' created successfully.")

    def get_retriever(self, config_key: str) -> Retriever:
        index_name = self._config_key_to_index_name.get(config_key)
        if not index_name or index_name not in self.retrievers:
            raise ValueError(f"Retriever for config key '{config_key}' not found")
        return self.retrievers[index_name]

    def list_retrievers(
        self,
        include_search_args: bool = False,
        index_as_main_key: bool = False,
    ) -> str:
        retriever_strings = []

        for index_name, retriever in self.retrievers.items():
            config_key = self._index_name_to_key.get(index_name, index_name)
            config = self._index_name_to_config.get(
                index_name, {}
            )  # WARNING: This returns a reference! Avoid mutating directly!

            # Clean unwanted fields
            if not include_search_args:
                config = dict(config)  # Copy so we don't mutate original
                config.pop("search_kwargs", None)

            index = config.get("index_name", index_name)
            description = config.get("description", "").strip().replace("\n", " ")
            node_label = config.get("node_label", "Unknown")

            if index_as_main_key:
                retriever_str = (
                    f"- Index: {index}\n"
                    f"  Node label: {node_label}\n"
                    f"  Description: {description}"
                )
            else:
                retriever_str = (
                    f"- Name: {config_key}\n"
                    f"  Index: {index}\n"
                    f"  Description: {description}"
                )

            retriever_strings.append(retriever_str)

        return "\n\n".join(retriever_strings)
