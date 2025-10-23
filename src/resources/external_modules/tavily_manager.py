import yaml
import os
from typing import List, Literal
from pydantic import SecretStr
from tavily import TavilyClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from src.resources.utils.time import fetch_utc_date_from_google
from src.resources.prompt_builder.base import PromptBuilder


class WebContentSummary(BaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(
        description="Important quotes and excerpts from the content"
    )


class TavilyWebSearchManager:
    """Singleton-style manager for Tavily client, summarization model, and search pipeline."""

    _tavily_client: TavilyClient | None = None
    _summarization_model: BaseChatModel | None = None
    _prompt_builder: PromptBuilder | None = None
    _web_summary_prompt: dict | None = None

    @classmethod
    def set_tavily_client(cls, client: TavilyClient):
        cls._tavily_client = client

    @classmethod
    def get_tavily_client(cls) -> TavilyClient:
        if cls._tavily_client is None:
            raise RuntimeError(
                "Tavily client not initialized. Call set_tavily_client first."
            )
        return cls._tavily_client

    @classmethod
    def set_summarization_model(cls, model: BaseChatModel, prompt_yaml_path: str):
        cls._summarization_model = model
        cls._prompt_builder = PromptBuilder()
        if cls._web_summary_prompt is None:
            with open(prompt_yaml_path, "r", encoding="utf-8") as f:
                cls._web_summary_prompt = yaml.safe_load(f)

    @classmethod
    def get_summarization_model(cls) -> BaseChatModel:
        if cls._summarization_model is None:
            raise RuntimeError(
                "Summarization model not initialized. Call set_summarization_model first."
            )
        return cls._summarization_model

    @staticmethod
    def _get_llm_for_provider(
        provider: str, model_name: str, model_args: dict, model_secrets: dict
    ) -> BaseChatModel:
        """
        LLM factory that reads API keys from the global-settings model-secrets.
        NOTE: This is copied from the LangChainModelFactory class for ease of use.
        """
        api_key_env_var = model_secrets.get(provider, {}).get("api_key_env_var")
        api_key = None
        if api_key_env_var:
            api_key = SecretStr(os.environ.get(api_key_env_var))

        if provider == "openai":
            kwargs = {"model": model_name, **model_args}
            if api_key:
                kwargs["api_key"] = api_key
            return ChatOpenAI(**kwargs)

        elif provider == "ollama":
            base_url = model_args.get("base_url", "http://localhost:11434")
            num_predict = model_args.get("num_predict", -1)
            return ChatOllama(
                model=model_name, base_url=base_url, num_predict=num_predict
            )

        elif provider == "google":
            kwargs = {"model": model_name, "max_tokens": None, **model_args}
            if api_key:
                kwargs["google_api_key"] = api_key
            return ChatGoogleGenerativeAI(**kwargs)

        else:
            raise ValueError(f"Provider specified is not supported: {provider}")

    @classmethod
    async def summarize_webpage_content(cls, webpage_content: str) -> str:
        """Summarize webpage content using the configured summarization model."""
        try:
            model = cls.get_summarization_model()
            structured_model = model.with_structured_output(WebContentSummary)

            if cls._web_summary_prompt is None:
                raise RuntimeError(
                    "Web summary prompt not initialized. Call set_summarization_model first."
                )

            if not cls._prompt_builder:
                raise RuntimeError(
                    "Failed to find an initialized prompt builder for the Tavily web search client"
                )

            prompt_as_messages = cls._prompt_builder.build_from_yaml(
                cls._web_summary_prompt,
                format_args={
                    "webpage_content": webpage_content,
                    "date": await fetch_utc_date_from_google(),
                },
            )
            summary = await structured_model.ainvoke(prompt_as_messages)

            return (
                f"<summary>\n{summary.summary}\n</summary>\n\n"
                f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
            )
        except Exception as e:
            print(f"Failed to summarize webpage: {str(e)}")
            return (
                webpage_content[:1000] + "..."
                if len(webpage_content) > 1000
                else webpage_content
            )

    @classmethod
    def deduplicate_search_results(cls, search_results: List[dict]) -> dict:
        unique_results = {}
        for response in search_results:
            for result in response["results"]:
                url = result["url"]
                if url not in unique_results:
                    unique_results[url] = result
        return unique_results

    @classmethod
    async def process_search_results(cls, unique_results: dict) -> dict:
        summarized_results = {}
        for url, result in unique_results.items():
            if result.get("raw_content"):
                content = await cls.summarize_webpage_content(result["raw_content"])
            else:
                content = result["content"]
            summarized_results[url] = {"title": result["title"], "content": content}
        return summarized_results

    @classmethod
    def tavily_search_multiple(
        cls,
        search_queries: List[str],
        max_results: int = 3,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
    ) -> List[dict]:
        """Perform search using Tavily API for multiple queries."""
        client = cls.get_tavily_client()
        search_docs = []
        for query in search_queries:
            result = client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            search_docs.append(result)
        return search_docs

    @classmethod
    def format_search_output(cls, summarized_results: dict) -> str:
        if not summarized_results:
            return "No valid search results found."
        formatted_output = "Search results:\n\n"
        for i, (url, result) in enumerate(summarized_results.items(), 1):
            formatted_output += (
                f"\n\n--- SOURCE {i}: {result['title']} ---\n"
                f"URL: {url}\n\n"
                f"SUMMARY:\n{result['content']}\n\n" + "-" * 80 + "\n"
            )
        return formatted_output
