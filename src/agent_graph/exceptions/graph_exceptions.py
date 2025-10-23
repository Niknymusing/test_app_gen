"""
This module holds all LangGraph graph-specific custom exceptions definitions.
We use custom exceptions for our graph for better readability, granular error handling and
cleaner logs.
"""


class BaseGraphException(Exception):
    """Base exception for LangGraph graph-specific errors."""

    def __init__(self, message: str, *, context: dict = None):
        super().__init__(message)
        self.context = context or {}


class GraphBuildException(BaseGraphException):
    """Exception raised when building the workflow graph fails."""

    def __init__(self, message: str, *, context: dict = None):
        super().__init__(message, context=context)


class ModelCallException(BaseGraphException):
    """Raised when a model call (e.g., LLM invocation) fails."""

    def __init__(self, message: str, *, context: dict = None):
        super().__init__(message, context=context)
