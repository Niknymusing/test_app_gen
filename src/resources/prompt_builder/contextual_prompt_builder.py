from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseContextRetriever(ABC):
    """
    Abstract base class for all context retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve context relevant to the query.

        :param query: The query to retrieve context for.
        :param kwargs: Additional arguments for the retriever.
        :return: A list of context documents, each as a dictionary.
        """
        pass


class ContextualPromptBuilder:
    """
    Builds prompts by orchestrating various context retrievers.
    """

    def __init__(self, retrievers: List[BaseContextRetriever]):
        self.retrievers = retrievers

    def build(self, base_prompt: str, query: str) -> str:
        """
        Build a contextual prompt.

        :param base_prompt: The base prompt template.
        :param query: The user query.
        :return: The final prompt with retrieved context.
        """
        all_context = []
        for retriever in self.retrievers:
            all_context.extend(retriever.retrieve(query))
        
        # Simple context formatting for now
        formatted_context = "\\n\\n".join(
            [f"{doc['source']}:\\n{doc['content']}" for doc in all_context]
        )

        return formatted_context
