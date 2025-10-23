"""
Common utility function around building prompts
"""

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    filter_messages,
    AnyMessage,
)
from typing import (
    List,
    Union,
    Literal,
    Sequence,
    Dict,
    Any,
    cast,
    Optional,
    Union,
    Tuple,
)
from langchain_core.runnables.base import Runnable
from langchain_core.messages.utils import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
import logging
from langchain_core.messages.utils import convert_to_messages
from langchain_core.language_models.chat_models import BaseChatModel


def validate_message_format_consistency(
    extra_messages: Union[List[AnyMessage], List[dict], dict, None],
) -> Tuple[Union[List[AnyMessage], List[Dict]], bool]:
    """
    Validates the format of the `extra_messages` and adjusts the conversation to OpenAI-style dictionaries
    accordingly.

    Returns:
        (extra_messages, as_open_ai_messages)

    Raises:
        TypeError or ValueError for invalid/mixed input.
    """
    if extra_messages is None:
        return [], False

    if isinstance(extra_messages, dict):
        raise TypeError(
            "`extra_messages` must be a list, not a single dict. Wrap it in a list."
        )

    if not isinstance(extra_messages, list):
        raise TypeError(
            "`extra_messages` must be a list of dicts or BaseMessage instances."
        )

    all_dicts = all(isinstance(m, dict) for m in extra_messages)
    all_base_messages = all(isinstance(m, BaseMessage) for m in extra_messages)

    if all_dicts:
        return extra_messages, True
    elif all_base_messages:
        return extra_messages, False
    else:
        raise ValueError(
            "`extra_messages` must be a list of either all Langchain `BaseMessage` instances "
            "or all OpenAI-style `dict`s, not a mix."
        )


def safe_convert_to_base_msg_list(
    messages: Union[
        List[Union[BaseMessage, Dict[str, Any]]],
        BaseMessage,
    ],
) -> List[BaseMessage]:
    """
    Convert input to a list of BaseMessage instances.

    Args:
        messages: A single BaseMessage, a list of BaseMessages, or a list of dict-like
                  message representations.

    Returns:
        A list containing BaseMessage objects.

    Raises:
        ValueError: If input cannot be converted to BaseMessage list.
    """
    try:
        if isinstance(messages, list):
            if messages and all(isinstance(m, BaseMessage) for m in messages):
                return cast(
                    List[BaseMessage], messages
                )  # We know we have a list of BaseMessages...
            else:
                # Attempt to convert list of dict-like to BaseMessage
                converted = convert_to_messages(messages)
                if not all(isinstance(m, BaseMessage) for m in converted):
                    raise ValueError(
                        "convert_to_messages returned items that are not BaseMessage"
                    )
                return converted
        elif isinstance(messages, BaseMessage):
            return [messages]
        else:
            # Single dict-like or unknown, try converting wrapped in list
            converted = convert_to_messages([messages])
            if not all(isinstance(m, BaseMessage) for m in converted):
                raise ValueError(
                    "convert_to_messages returned items that are not BaseMessage"
                )
            return converted
    except Exception as e:
        raise ValueError(f"Failed to convert messages to BaseMessage list: {e}") from e


def safe_append_to_base_msg_list(
    messages: Union[List[Union[BaseMessage, Dict[str, Any]]], BaseMessage, None],
    new_messages: Union[
        BaseMessage, Dict[str, Any], List[Union[BaseMessage, Dict[str, Any]]]
    ],
    keep_only_oldest_system: bool = False,
) -> List[BaseMessage]:
    """
    Ensure messages is a list of BaseMessage, convert new_messages (single or list of BaseMessage or OpenAI dict)
    to BaseMessage, and append them.

    Args:
        messages: Existing messages list or single message (BaseMessage or dict-like).
        new_messages: New message(s) as BaseMessage, OpenAI-style dict, or list thereof.
        keep_only_oldest_system: If True, keeps only the oldest system message

    Returns:
        List[BaseMessage]: A new list with new_messages appended.

    Raises:
        ValueError: If conversion fails or new_message format is unknown.
    """

    # Verify the format of the existing messages list first...
    base_messages = (
        safe_convert_to_base_msg_list(messages) if messages is not None else []
    )

    # Then we conver new messages...
    if not isinstance(new_messages, list):
        new_messages = [new_messages]
    new_messages_list = safe_convert_to_base_msg_list(new_messages)

    all_msgs = base_messages + new_messages_list

    if keep_only_oldest_system:
        oldest_system = None
        non_system_msgs = []
        for msg in all_msgs:
            if isinstance(msg, SystemMessage) and oldest_system is None:
                oldest_system = msg
            elif not isinstance(msg, SystemMessage):
                non_system_msgs.append(msg)
        all_msgs = ([oldest_system] if oldest_system else []) + non_system_msgs

    return all_msgs


def prepend_latest_system_message(
    messages: Sequence[Union[BaseMessage, dict]],
    format: Literal["langchain", "openai"] = "langchain",
) -> List[Union[BaseMessage, dict]]:
    """
    Ensure only the latest system message is kept and prepended to the list.

    Args:
        messages: List of messages, either LangChain BaseMessage objects or OpenAI-style dicts.
        format: Indicates message format; "langchain" or "openai".

    Returns:
        Updated list with only the latest system message prepended.
    """
    latest_system_message = None

    if format == "langchain":
        for msg in reversed(messages):
            if isinstance(msg, SystemMessage):
                latest_system_message = msg
                break

        non_system_messages = [
            msg for msg in messages if not isinstance(msg, SystemMessage)
        ]

    elif format == "openai":
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                latest_system_message = msg
                break

        non_system_messages = [
            msg
            for msg in messages
            if not (isinstance(msg, dict) and msg.get("role") == "system")
        ]

    else:
        raise ValueError(
            f"Unknown format '{format}'. Expected 'langchain' or 'openai'."
        )

    if latest_system_message:
        return [latest_system_message] + non_system_messages
    else:
        return non_system_messages


def contains_system_message(messages: list) -> bool:
    """
    Check if a SystemMessage exists in the list of messages.
    Supports both LangChain BaseMessage objects and OpenAI-style dicts.
    """
    for message in messages:
        # LangChain message object
        if hasattr(message, "type") and message.type == "system":
            return True
        # OpenAI-style dict
        if isinstance(message, dict) and message.get("role") == "system":
            return True
    return False


def safe_trim_llm_message_list(
    messages_list: List[BaseMessage],
    llm: Union[BaseChatModel, Runnable],
    max_tokens_per_tool_msg: int = 10000,
    trim_tool_responses=True,
    logger: Optional[logging.Logger] = None,
):
    """
    Trims a list of Langchain BaseMessage instances to prevent exceeding token limits, keeping the first message (system prompt) and the most recent ones.
    Uses the LLM's token counting method if available, otherwise falls back to a basic approximation with a lambda function.
    """
    try:
        # Determine the token counting method
        if hasattr(llm, "get_num_tokens_from_messages"):
            token_counter = llm.get_num_tokens_from_messages  # type: ignore
        else:
            # Fallback: Basic approximation
            token_counter = count_tokens_approximately

        if trim_tool_responses:
            # Pre-trim tool message responses
            # NOTE: Sometimes when executing tools we can end up with n-million token long responses (e.g. querying databases), if we don't handle this
            #       we are risking filling the context window..For some reason the trimming function from langchain doesn't seem to handle
            #       tool messages very well so we need to handle it ourselves...
            CHARS_PER_TOKEN = 4.0  # same as in count_tokens_approximately

            for msg in messages_list:
                if isinstance(msg, ToolMessage):
                    approx_tokens = count_tokens_approximately([msg])
                    if logger:
                        logger.debug(
                            f"Tool message (ID: {msg.tool_call_id}) uses ~{approx_tokens} tokens"
                        )

                    if approx_tokens > max_tokens_per_tool_msg:
                        truncated_content = truncate_message_content(
                            content=str(msg.content),
                            max_tokens=max_tokens_per_tool_msg,
                            chars_per_token=CHARS_PER_TOKEN,
                        )
                        msg.content = truncated_content

                        # Re-count and log new number of tokens...
                        approx_tokens_new = count_tokens_approximately([msg])
                        if logger:
                            logger.debug(
                                f"Tool message (ID: {msg.tool_call_id}) contained approximately {approx_tokens} tokens before truncation "
                                f"and approximately {approx_tokens_new} tokens after truncation."
                            )

        # Trim messages
        trimmed_messages = trim_messages(
            messages_list,
            strategy="last",
            token_counter=token_counter,
            include_system=True,
            start_on=("human"),  # Always include the latest human message
            end_on=("human", "tool"),  # Always end with a human/tool message
        )

    except Exception as e:
        # In case of an error, return the original list
        trimmed_messages = messages_list

    return trimmed_messages


def truncate_message_content(
    content: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    truncation_marker: str = "...truncated...",
) -> str:
    """
    Truncate the message content approximately to a maximum number of tokens.

    Args:
        content: The message content string to truncate.
        max_tokens: Maximum tokens allowed.
        chars_per_token: Approximate chars per token (default 4).
        truncation_marker: String to append if truncation occurs.

    Returns:
        Truncated string with truncation_marker appended if truncated.
    """
    if not content:
        return ""

    approx_tokens = count_tokens_approximately([content])
    if approx_tokens <= max_tokens:
        return content

    max_chars = int(max_tokens * chars_per_token)
    # Ensure truncation marker fits in max_chars
    truncation_length = max_chars - len(truncation_marker)
    if truncation_length <= 0:
        # Not enough room for content, return just truncation marker
        return truncation_marker

    truncated = content[:truncation_length]
    return truncated + truncation_marker


def get_latest_human_ai_pairs(
    messages: List[BaseMessage],
    max_pairs: int = 8,
):
    """
    Extract the most recent conversation turns between a human and the AI.

    This helper filters out system messages and tool responses, then walks
    backwards through the conversation to collect up to `max_pairs` of
    (HumanMessage, AIMessage) pairs. The resulting conversation is returned
    either as a flat list of messages.

    Args:
        messages (List[BaseMessage]):
            Full conversation history including human, AI, system, and tool messages.
        max_pairs (int, optional):
            Maximum number of human/AI pairs to return. Defaults to 8.

    Returns:
        Union[List[BaseMessage]]:
            The extracted conversation history as a list of messages,

    Notes:
        - System messages (role="system") are removed.
        - Tool responses (`ToolMessage`) are ignored.
        - Pairs are returned in original chronological order.
    """

    def filter_out_system_messages(messages: list) -> list:
        """
        Helper to remove all system messages from the list.
        Supports both LangChain BaseMessage objects and OpenAI-style dicts.
        """
        return [
            msg
            for msg in messages
            if not (
                (hasattr(msg, "type") and msg.type == "system")
                or (isinstance(msg, dict) and msg.get("role") == "system")
            )
        ]

    # First filter out all system messages...
    messages = filter_out_system_messages(messages=messages)

    # We'll iterate backwards to find valid pairs from the end
    pairs = []
    i = len(messages) - 1

    while i > 0 and len(pairs) < max_pairs:
        current = messages[i]
        prev = messages[i - 1]

        # Check if prev is human (any BaseMessage but not AIMessage?), current is AIMessage
        if isinstance(current, AIMessage) and not isinstance(current, ToolMessage):
            if isinstance(prev, HumanMessage):
                # This is a valid human+AI pair (prev, current)...
                pairs.append((prev, current))
                i -= 2  # Skip the pair
                continue
        i -= 1

    # Pairs were collected backwards, reverse them to get original order
    pairs.reverse()

    # Flatten pairs to a list of messages again
    conversation_history = [msg for pair in pairs for msg in pair]

    return conversation_history


def get_tool_messages_contents(messages: list[BaseMessage]) -> list[str]:
    """Extract content from ToolMessage objects in a message history.

    This utility function scans through a sequence of messages and collects
    the content from any messages of type ToolMessage. It is useful when
    downstream logic needs to work only with the textual results or outputs
    that tools have returned during an interaction.

    Args:
        messages: List of message objects from a conversation history.

    Returns:
        List of strings containing the content from ToolMessage objects.
    """
    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]
