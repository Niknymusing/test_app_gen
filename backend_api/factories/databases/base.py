"""Factory interface for setting up database clients"""

from abc import ABC, abstractmethod
from typing import Dict
import logging
import os


class DBClientFactory(ABC):
    """Base class for building database clients using a factory pattern"""

    @staticmethod
    def get_env(env_vars_map: dict, key: str):
        """Small helper used to fetch env variables from the configuration set in 'agent_config.yaml'"""
        return os.getenv(env_vars_map.get(key, ""), None)

    @abstractmethod
    def create(
        self,
        db_config: Dict,
        logger: logging.Logger,
        model_secrets: Dict,
        client_name_key: str,
    ) -> Dict:
        """

        Returns:
        A dictionary, like as for example:
        {
            client: Neo4jClient
            embeddings_model: Optional[Embeddings]
        }
        """
        pass
