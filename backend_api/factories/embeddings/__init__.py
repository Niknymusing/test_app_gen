from .openai import OpenAIEmbeddingFactory


EMBEDDING_FACTORIES = {
    "openai": OpenAIEmbeddingFactory(),
}


def get_embedding_factory(provider: str):
    provider = provider.lower()
    factory = EMBEDDING_FACTORIES.get(provider)
    if not factory:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return factory
