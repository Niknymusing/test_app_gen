from src.resources.vector_database_clients.qdrant_client import QdrantClient
from backend_api.factories.databases.base import DBClientFactory
from backend_api.factories.embeddings import get_embedding_factory
from backend_api.factories.helpers import verify_model_secrets
from typing import Dict
import logging


class QdrantClientFactory(DBClientFactory):
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
        qdrant_url = self.get_env(env_vars_map, "qdrant_url")
        qdrant_port = int(self.get_env(env_vars_map, "qdrant_port") or "6333")

        # --- Check that we have all required vars ---
        if not qdrant_url:
            raise EnvironmentError(
                "Missing required environment variable for Qdrant connection: qdrant_url"
            )

        # Embeddings model initialization
        embedding_factory = get_embedding_factory(provider)
        embeddings_model = embedding_factory.initialize_embedding_model(
            embedding_cfg=embedding_cfg,
            model_secrets=model_secrets,
        )

        client = QdrantClient(
            embeddings_model=embeddings_model,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            logger=logger,
            retriever_configs=retriever_configs,
            client_name_key=client_name_key,
        )

        return {
            "client": client,
            "embedding_model": embeddings_model,
        }
