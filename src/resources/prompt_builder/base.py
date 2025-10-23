import re
from typing import List, Optional, Dict, Any, Union, Tuple, Dict
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    AnyMessage,
)
from resources.utils.messages import (
    prepend_latest_system_message,
    validate_message_format_consistency,
)
from langchain_core.messages.utils import convert_to_openai_messages
from .contextual_prompt_builder import ContextualPromptBuilder, BaseContextRetriever


class PromptBuilder:
    def __init__(self, context_retrievers: Optional[List[BaseContextRetriever]] = None):
        if context_retrievers:
            self.contextual_builder = ContextualPromptBuilder(context_retrievers)
        else:
            self.contextual_builder = None

    def build_from_yaml(
        self,
        prompt_yaml: Dict[str, Any],
        format_args: Optional[Dict[str, Any]] = None,
        extra_messages: Union[List[AnyMessage], List[dict]] = [],
        prepend_extra_messages: bool = True,
        as_open_ai_messages=False,
        query_for_context: Optional[str] = None,
    ) -> List[Union[BaseMessage, dict]]:
        """
        Build a list of messages from a YAML prompt definition and format_args.
        """

        if self.contextual_builder and query_for_context:
            # This is a simplified integration. A more robust implementation
            # would involve a more sophisticated way of incorporating context.
            context = self.contextual_builder.build("", query_for_context)
            if format_args:
                format_args['context'] = context
            else:
                format_args = {'context': context}

        if not isinstance(prompt_yaml, dict) or "messages" not in prompt_yaml:
            raise ValueError("Prompt YAML must be a dict with a 'messages' key")

        messages = []
        for mapping in prompt_yaml["messages"]:
            role = mapping["role"]
            # Check if this message uses template or content
            if "template" in mapping:
                template = mapping["template"]
                try:
                    content = template.format(**(format_args or {}))
                except KeyError:
                    # Some keys might be missing, which is fine.
                    # We can try to format with only the available keys.
                    keys = re.findall(r'{(\w+)}', template)
                    filtered_args = {k: v for k, v in (format_args or {}).items() if k in keys}
                    content = template.format(**filtered_args)
            elif "content" in mapping:
                content = mapping["content"]
            else:
                raise ValueError(
                    f"Message with role '{role}' has neither 'template' nor 'content' field"
                )

            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role in prompt YAML: {role}")

        # Validate the 'extra_messages' format
        extra_messages, as_open_ai_messages = validate_message_format_consistency(
            extra_messages
        )
        if as_open_ai_messages:
            messages = convert_to_openai_messages(messages)

        if prepend_extra_messages:
            combined_messages = (extra_messages or []) + messages  # type: ignore
        else:
            combined_messages = messages + (extra_messages or [])  # type: ignore

        # Use the helper function to prepend the latest system message (if any/keep only one)
        messages = prepend_latest_system_message(
            combined_messages, format="openai" if as_open_ai_messages else "langchain"
        )

        return messages
