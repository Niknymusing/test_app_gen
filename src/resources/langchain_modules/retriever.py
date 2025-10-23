"""
Retrieval components for the RAG system.
"""

from typing import Dict, Any, Optional
from langchain_core.vectorstores import VectorStore


class Retriever:
    """Class for handling document retrieval."""

    def __init__(
        self,
        knowledge_base: VectorStore,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the retriever.

        Args:
            knowledge_base: The vector store to retrieve from
            search_kwargs: Search parameters for the retriever.
        """

        self.search_kwargs = search_kwargs or {
            "score_threshold": 0.5,
            "k": 4,
        }  # K refers to the number of top results to be returned on retrieval
        self.knowledge_base = knowledge_base
        self.retriever = self._init_retriever()

    def _init_retriever(self):
        """Initializes the base retriever and optionally wraps it with an LLM-based context compressor."""
        base_retriever = self.knowledge_base.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=self.search_kwargs
        )
        return base_retriever

    async def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on the question.

        Args:
            question: The question to retrieve documents for

        Returns:
            Dictionary with context, few-shot examples
        """
        try:
            docs = await self.retriever.ainvoke(question)
        except Exception as e:
            return []

        return docs

    async def __call__(self, question: str) -> Dict[str, Any]:
        """
        Make the retriever directly callable.

        Args:
            question: The question to retrieve documents for

        Returns:
            Dictionary with context, few-shot examples
        """
        return await self.retrieve(question)
