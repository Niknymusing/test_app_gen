"""
Utility functions around the use of tools
"""

from typing import List, Optional, Union
from langchain_core.messages import AIMessage, BaseMessage, AnyMessage


def route_to_tools_or_next(
    next_node_name: str,
    tools_node_name: str = "tools",
    messages_list: Optional[Union[List[AnyMessage], List[BaseMessage]]] = None,
) -> str:
    """
    Decides whether to route to the tools node or the next node based on the presence
    of tool calls in the last AIMessage within the provided messages list.

    If no tool calls are found, routing continues to the provided `next_node_name`.

    Args:
        next_node_name (str): The node to route to if no tool calls are present.
        tools_node_name (str, optional): The node to route to when tool calls exist. Defaults to "tools".
        messages_list (Optional[Union[List[AnyMessage], List[BaseMessage]]], optional):
            List of messages to inspect for tool calls. Defaults to None.

    Returns:
        str: The name of the next node to route to.
    """
    if not messages_list:
        return next_node_name

    last_message = messages_list[-1]
    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        if last_message.tool_calls:
            return tools_node_name
    return next_node_name


def get_latest_ai_tool_msg_chain(messages: List[AnyMessage]):
    """
    Returns the latest AI message with tool calls and all subsequent tool response messages
    matching those calls.

    Useful for agent-tool interactions where one AI message issues multiple tool calls,
    and multiple tool responses follow. Returns an empty list if no such pair is found.

    Args:
        messages (List[BaseMessage]): Chronological list of messages including AI and tool messages.

    Returns:
        List[BaseMessage]: The latest AI message with tool calls and its corresponding tool responses.
    """
    ai_idx = None
    tool_call_ids = set()

    # 1. Find the latest AI message with tool_calls
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            ai_idx = idx
            tool_call_ids = {tc["id"] for tc in msg.tool_calls}
            break

    # 2. Collect all tool messages that match the tool_call_ids
    if ai_idx is not None and tool_call_ids:
        tool_msgs = [
            msg
            for msg in messages[ai_idx + 1 :]
            if getattr(msg, "type", None) == "tool"
            and getattr(msg, "tool_call_id", None) in tool_call_ids
        ]

        if tool_msgs:
            return [messages[ai_idx]] + tool_msgs

    return []


def format_tools_for_prompt(tools_dict: dict):
    """
    Takes a dictionary of tools and returns a formatted string
    listing each tool's name and description.
    """
    formatted_entries = []

    for tool_key, tool_value in tools_dict.items():
        # Extract the tool name and description
        tool_name = tool_value.name
        tool_description = tool_value.description
        formatted_entries.append(f"- **{tool_name}**: {tool_description}")

    return "\n".join(formatted_entries)


def map_tool_calls_by_tool_name(
    tool_calls: list[dict], names: list[str]
) -> dict[str, list[dict]]:
    """
    Categorize tool calls by their tool names.

    Args:
        tool_calls: List of tool call dicts, typically from an AIMessage.
        names: List of tool call names to separate.

    Returns:
        Dict mapping each name to a list of tool calls with that name.

    Example:
        >>> tool_calls = [
        ...     {"name": "search", "args": {"query": "weather"}},
        ...     {"name": "summarize", "args": {"text": "long article"}},
        ...     {"name": "search", "args": {"query": "news"}}
        ... ]
        >>> map_tool_calls_by_tool_name(tool_calls, ["search", "summarize"])
        {
            "search": [
                {"name": "search", "args": {"query": "weather"}},
                {"name": "search", "args": {"query": "news"}}
            ],
            "summarize": [
                {"name": "summarize", "args": {"text": "long article"}}
            ]
        }
    """
    categorized = {name: [] for name in names}
    for call in tool_calls:
        if call["name"] in categorized:
            categorized[call["name"]].append(call)
    return categorized
