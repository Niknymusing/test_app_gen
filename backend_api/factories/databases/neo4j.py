from src.resources.vector_database_clients.neo4j_client import Neo4jClient
from backend_api.factories.databases.base import DBClientFactory
from backend_api.factories.embeddings import get_embedding_factory
from backend_api.factories.helpers import verify_model_secrets
from typing import Dict
import logging
from src.resources.tools.cypher import (
    set_neo4j_client_for_tools,
    set_embedding_model_for_tools,
)


class Neo4jClientFactory(DBClientFactory):
    def create(
        self,
        db_config: Dict,
        logger: logging.Logger,
        model_secrets: Dict,
        client_name_key: str,
    ) -> Dict:
        # -- Read configurations ---
        env_vars_map = db_config.get("env_vars", {})
        embedding_cfg = db_config.get("embedding_model", {})
        retriever_configs = db_config.get("retriever_configs", {})
        provider = embedding_cfg.get("provider", "").lower()

        # --- Verify model secrets/provider ---
        verify_model_secrets(
            provider=provider, model_secrets=model_secrets, logger=logger
        )

        # -- Read all required environment variables from config ---
        neo4j_uri = self.get_env(env_vars_map, "neo4j_uri")
        neo4j_username = self.get_env(env_vars_map, "neo4j_username")
        neo4j_password = self.get_env(env_vars_map, "neo4j_password")
        neo4j_database = self.get_env(env_vars_map, "neo4j_database") or "neo4j"

        # --- Check that we have all required vars ---
        missing = [
            k
            for k, v in {
                "neo4j_uri": neo4j_uri,
                "neo4j_username": neo4j_username,
                "neo4j_password": neo4j_password,
            }.items()
            if not v
        ]

        if missing:
            missing_vars = ", ".join(missing)
            raise EnvironmentError(
                f"Missing required environment variables for Neo4j connection: {missing_vars}"
            )

        # Embeddings model initialization
        embedding_factory = get_embedding_factory(provider)
        embeddings_model = embedding_factory.initialize_embedding_model(
            embedding_cfg=embedding_cfg,
            model_secrets=model_secrets,
        )

        client = Neo4jClient(
            embeddings_model=embeddings_model,
            neo4j_url=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            logger=logger,
            retriever_configs=retriever_configs,
            client_name_key=client_name_key,
        )

        # -- Share the initilized embedding model / Neo4j client with the Cypher query execution tool--
        set_embedding_model_for_tools(embeddings_model)
        set_neo4j_client_for_tools(client)

        return {
            "client": client,
            "embedding_model": embeddings_model,
        }
