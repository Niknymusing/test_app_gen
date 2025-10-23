import logging
import os


def verify_model_secrets(provider: str, model_secrets: dict, logger: logging.Logger):
    if not provider:
        raise ValueError("Embedding provider must be specified in the configuration.")

    if model_secrets is None or provider not in model_secrets:
        msg = f"Missing model secrets for embedding provider '{provider}'."
        logger.error(msg)
        raise EnvironmentError(msg)

    provider_secrets = model_secrets.get(provider, {})
    api_key_env_var = provider_secrets.get("api_key_env_var")

    if not api_key_env_var or not os.getenv(api_key_env_var):
        msg = (
            f"API key environment variable '{api_key_env_var}' for provider '{provider}' "
            "is not set or missing."
        )
        logger.error(msg)
        raise EnvironmentError(msg)
