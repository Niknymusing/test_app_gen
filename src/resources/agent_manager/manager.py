"""
Manager class for managing sub-agents to a master agent
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_graph.graph import AgentGraph
from typing import List, Optional
import logging


class SubagentManager:
    """
    A manager for registering AgentGraph instances and providing descriptions for prompt injection.

    This class:
    1. Stores AgentGraph instances and their descriptions
    2. Provides formatted information for prompt injection
    3. Allows access to compiled graphs for building master agents
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the SubagentManager.

        Args:
            logger: Optional logger for tracking subagent operations
        """
        self.subagents = {}
        self.subagent_descriptions = {}
        self.subagent_modes = {}
        self.logger = logger or logging.getLogger(__name__)
        self.valid_subagent_modes = ["callable_function", "node"]

    def register_agent(
        self,
        name: str,
        agent_graph: "AgentGraph",
        description: str = "",
        create_as: str = "node",
    ):
        """
        Register a subagent with the manager.

        Args:
            name: Name of the subagent
            agent_graph: The AgentGraph instance
            description: Description of what the subagent does
            create_as: Configuration on how the subagent will be used/created inside the parent agent
        """
        if create_as not in self.valid_subagent_modes:
            raise ValueError(
                f"Invalid usage mode configuration for subagent {name} found with value {create_as}. Valid options are: {self.valid_subagent_modes}"
            )
        self.subagents[name] = agent_graph
        self.subagent_descriptions[name] = description
        self.subagent_modes[name] = create_as

        if self.logger:
            self.logger.info(f"Registered subagent: {name}")
            self.logger.debug(
                f"  Usage in parent graph as: {create_as}\n  Description: {description}"
            )

    def get_compiled_graph(self, name: str):
        """
        Get the compiled graph of a registered subagent.

        Args:
            name: Name of the subagent

        Returns:
            The compiled graph of the subagent, or None if not found
        """
        agent = self.subagents.get(name)
        if agent and hasattr(agent, "graph"):
            return agent.graph
        return None

    def list_agents(self) -> List[str]:
        """List all registered subagent names."""
        return list(self.subagents.keys())

    def get_agents_for_prompt(self) -> str:
        """
        Get a formatted string describing all available subagents for prompt injection.

        Returns:
            A formatted string with agent names and descriptions
        """
        if not self.subagents:
            return "No subagents are currently available."

        result = "Available subagents:\n\n"
        for name, agent in self.subagents.items():
            description = self.subagent_descriptions.get(
                name, "No description available"
            )
            result += f"- {name}: {description}\n"

        result += "\nTo use a subagent, specify its name in your response."
        return result
