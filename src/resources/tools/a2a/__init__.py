"""
A2A (Agent-to-Agent) Communication Tools for MEU Framework

This module provides tools for agent-to-agent communication using the A2A protocol,
enabling collaboration between MEU agents and other specialized agents.
"""

from .a2a_tools import (
    discover_agents,
    send_message,
    request_collaboration,
    share_task_result,
    query_agent_capabilities
)

__all__ = [
    "discover_agents",
    "send_message",
    "request_collaboration",
    "share_task_result",
    "query_agent_capabilities"
]