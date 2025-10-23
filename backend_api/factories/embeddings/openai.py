from backend_api.factories.embeddings.base import BaseEmbeddingFactory
from langchain_openai import OpenAIEmbeddings
import os
from pydantic import SecretStr


class OpenAIEmbeddingFactory(BaseEmbeddingFactory):
    def initialize_embedding_model(
        self, embedding_cfg: dict, model_secrets: dict = {}
    ) -> OpenAIEmbeddings:
        model_name = embedding_cfg.get("model", "text-embedding-3-large")
        embedding_dims = int(embedding_cfg.get("dims", 1536))

        api_key = ""
        if model_secrets:
            openai_secrets = model_secrets.get("openai", {})
            api_key_env_var = openai_secrets.get("api_key_env_var")
            if not api_key_env_var:
                raise ValueError(
                    "Missing 'api_key_env_var' for OpenAI in model-secrets config"
                )
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise EnvironmentError(
                    f"Environment variable '{api_key_env_var}' for OpenAI API key is not set."
                )

        return OpenAIEmbeddings(
            model=model_name,
            dimensions=embedding_dims,
            api_key=SecretStr(api_key),
        )
