"""
Long-term memory component built with mem0 for storing/accessing memories from a Qdrant Vector Database
"""

import logging
from typing import Dict, List, Union, Optional
from mem0 import AsyncMemory


class AsyncMem0Memory:
    """
    Simple memory component using mem0 for long-term storage and retrieval of conversations.
    """

    def __init__(
        self,
        collection_name: str,
        host: str,
        port: int,
        api_key: str,
        llm_model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-3-large",
        embedding_dims: int = 1536,  # Default for "text-embedding-3-large"
        llm_temperature: float = 0.2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the memory component with explicit parameters.

        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant host address
            port: Qdrant port number
            api_key: OpenAI API key
            llm_model: LLM model name (default: gpt-4.1)
            llm_temperature: Temperature for LLM operations (default: 0.2)
            embedding_model: Embedding model name (default: text-embedding-3-large)
            logger: Optional logger instance

        Raises:
            Exception: If memory initialization fails
        """
        self.logger = logger or logging.getLogger(__name__)

        # Store configuration
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.temperature = llm_temperature

        # Initialize memory
        self.memory = None

    async def aconfigure(self):
        """Asynchronous initializer for the memory component"""
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "host": self.host,
                    "port": self.port,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    "api_key": self.api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "api_key": self.api_key,
                    "embedding_dims": self.embedding_dims,
                },
            },
        }

        self.memory = await AsyncMemory.from_config(config)
        self.logger.info("✅ Long-term memory configured and ready")
        return self

    async def add(
        self,
        messages: Union[str, Dict, List[Dict]],
        user_id: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Add messages to memory.

        Args:
            messages: Messages to store in memory
            user_id: User identifier to associate with these memories
            metadata: Optional metadata to store with these memories

        Returns:
            Result of the memory addition operation

        Raises:
            Exception: If adding to memory fails
        """
        try:
            self.logger.info(f"Updating memory for user with ID: {user_id}")

            # Call the memory.add method
            result = await self.memory.add(
                messages=messages, user_id=user_id, metadata=metadata or {}
            )

            # Log based on the result
            if result and "results" in result and result["results"]:
                # Count operations by type
                operation_counts = {}
                for item in result["results"]:
                    if "event" in item:
                        event_type = item["event"]
                        operation_counts[event_type] = (
                            operation_counts.get(event_type, 0) + 1
                        )

                # Create a summary of operations
                if operation_counts:
                    operations_summary = ", ".join(
                        [
                            f"{count} {op_type}"
                            for op_type, count in operation_counts.items()
                        ]
                    )
                    self.logger.info(
                        f"Memory operations for user with ID {user_id}: {operations_summary}"
                    )
                else:
                    self.logger.info(
                        f"Memory operations performed but no event types found for user with ID: {user_id}"
                    )
            else:
                # No memory operations were performed
                self.logger.info(
                    f"No memory operations were performed for user with ID: {user_id}"
                )
                return result

        except Exception as e:
            error_message = f"Error adding memory: {str(e)}"
            self.logger.error(error_message)
            raise  # Re-raise the exception

    async def search(self, query: str, user_id: str, limit: int = 5) -> str:
        """
        Search for memories relevant to a query.

        Args:
            query: Search query text
            user_id: User ID to filter memories by
            limit: Maximum number of results to return

        Returns:
            Search results as a string

        Raises:
            Exception: If searching memory fails
        """
        try:
            user_info = (
                f"for user with ID: {user_id}" if user_id else "(no user ID provided)"
            )
            self.logger.info(f"Searching memories with query: {query} {user_info}")

            # Search for memories
            results = await self.memory.search(
                query=query, user_id=user_id, limit=limit
            )

            return self.convert_memories_to_str(results, limit, user_id)
        except Exception as e:
            error_message = (
                f"Error encountered when searching for relevant memories: {str(e)}"
            )
            self.logger.error(error_message)
            raise  # Re-raise the exception

    def convert_memories_to_str(
        self, memories_result: Dict, max_memories: int = 5, user_id: str = ""
    ) -> str:
        """
        Convert memory search results to a formatted string for inclusion in prompts.

        Args:
            memories_result: The result from memory.search()
            max_memories: Maximum number of memories to include
            user_id: The user's string unique ID

        Returns:
            Formatted string with memories or a message indicating no memories found
        """
        # Check if we have results
        missing_memories_msg = (
            "No relevant memories from past interactions with the user found."
        )
        if not memories_result:
            return missing_memories_msg

        # Get the results list
        results = memories_result.get("results", [])

        # If results is empty, return default message
        if not results:
            return missing_memories_msg

        # Log the number of retrieved facts from memories
        self.logger.info(
            f"Retrieved {len(results)} relevant memories for the request made by user with ID: {user_id}"
        )

        memories = []
        for idx, memory in enumerate(results[:max_memories], 1):
            # Extract the memory content - in your format it's directly in 'memory' field
            if "memory" in memory:
                content = memory["memory"]
                memories.append(f"Fact {idx} for user: {content}")

        # Combine all memories with separators (without the header)
        if memories:
            return "\n\n".join(memories)
        else:
            return missing_memories_msg
