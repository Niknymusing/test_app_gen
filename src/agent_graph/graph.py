"""
Generic LangGraph agent class
"""

import logging
import os
import json
import yaml
import inspect
import importlib
import functools
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Type, Union, cast
from langgraph.graph import StateGraph
from langchain_core.runnables.base import Runnable
from langchain_core.tools import BaseTool
from pathlib import Path
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import StreamWriter
from src.agent_graph.exceptions.graph_exceptions import (
    GraphBuildException,
    ModelCallException,
)

from src.resources.prompt_builder.base import PromptBuilder
from src.resources.utils.messages import safe_trim_llm_message_list
from src.resources.external_modules.mem0_memory import AsyncMem0Memory
from langchain_core.output_parsers import StrOutputParser
from src.resources.agent_manager.manager import SubagentManager
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from src.resources.vector_database_clients.base import BaseVectorDBClient
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode


class AgentGraph:
    """Class for building and running the RAG workflow graph."""

    def __init__(
        self,
        config: Dict,
        agent_name: str,
        llm_map: Dict[str, Union[BaseChatModel, Runnable]],
        llm_tools_map: Optional[Dict[str, List[BaseTool]]] = None,
        logger: Optional[logging.Logger] = None,
        memory_saver: Optional[BaseCheckpointSaver] = None,
        subagent_manager: Optional[SubagentManager] = None,
        vector_db_clients: Optional[Dict[str, BaseVectorDBClient]] = None,
        is_master_agent: bool = False,
    ):
        """
        Initialize the graph builder.

        Args:
            config: The loaded configuration file's contents
            agent_name: The name of the agent to build (should match the one in the 'agent_config.yaml')
            llm_map: The LLM (optionally with bound tools) mappings per-node that will be used in the pipeline
            llm_tools_map(Optional): A mapping of per-node tool names to their corresponding LLM tool instances (Can be ommited if we don't use tools)
            memory_saver: The initialized memory saver (if any)
            subagent_manager: Optional SubagentManager for multi-agent architectures
            vector_db_clients: A name,database client mapping of ALL initializated database clients across the application(will be access filtered for each agent individually)
            is_master_agent: Flag used to keep track of which agent is the 'master' agent. We need this to handle memory configuration(subagents will not use any memory!).
        """
        # Load the config file
        self.config = config
        self.logger = logger

        # Resources and graph components
        self.llm_map = llm_map
        self.agent_name = agent_name
        self.prompt_builder = PromptBuilder()
        self.memory = memory_saver or InMemorySaver()
        self.use_memory = is_master_agent and self.config.get("agents", {}).get(
            agent_name, {}
        ).get("settings", {}).get("use_memory", False)
        self.long_term_memory_enabled = (
            "long-term-memory"
            in self.config.get("agents", {}).get(agent_name, {}).get("settings", {})
            and self.use_memory
        )
        self.long_term_memory = None

        # NOTE: Just a LangGraph native in-memory store that can be used to store stuff in-memory during interactions
        # Should mainly be used for development and not production!
        self.in_memory_store = InMemoryStore()

        # Get allowed database connections for this agent from config
        agent_settings = (
            self.config.get("agents", {}).get(agent_name, {}).get("settings", {})
        )
        allowed_dbs = agent_settings.get("available_databases", [])

        if not isinstance(allowed_dbs, (list, tuple, set)):
            if self.logger:
                self.logger.warning(
                    f"'available_databases' for agent '{agent_name}' is not a list; defaulting to empty list."
                )
            allowed_dbs = []

        if not isinstance(vector_db_clients, dict):
            if self.logger:
                self.logger.warning(
                    f"Expected vector_db_clients to be a dict, got {type(vector_db_clients)}. Defaulting to empty dict."
                )
            vector_db_clients = {}

        # A dictionary holding all the database clients that the agent is allowed to use (if any)
        # A client can be used to connect or interact with a database (e.g. get schema for Neo4) and
        # contains different Retriever classes according to the setup on the 'agent_config' YAML file.
        # NOTE: This is propageted to node functions to allow access to different retrievers.

        # --- Filter clients by allowed DBs ---
        filtered_clients = {
            name: client
            for name, client in vector_db_clients.items()
            if name in allowed_dbs
        }

        # Log if some allowed DB names are missing in the provided clients
        missing_dbs = set(allowed_dbs) - set(vector_db_clients.keys())
        if missing_dbs and self.logger:
            self.logger.warning(
                f"Agent '{agent_name}' requested unavailable DB clients: {missing_dbs}"
            )
        self.vector_db_clients = filtered_clients

        # Utility class that holds initialized subagents with their descriptions
        # We can use this to create a multi-agent system and connect those to a parent agent's workflow
        # during build()
        self.subagent_manager = subagent_manager if subagent_manager else None

        # NOTE: This is used to store the detailed tool map for each node of an agent
        self.tools_map_per_node = llm_tools_map or {}

        # NOTE: This holds the dedicated ToolNode instances per graph node, each
        #       configured with the right tools.
        self.tool_nodes_per_node = {}

        self.graph = None

    @classmethod
    async def acreate(
        cls,
        config: Dict,
        agent_name: str,
        llm: Dict[str, Union[BaseChatModel, Runnable]] = {},
        llm_tools_map: Dict[str, List[BaseTool]] = {},
        logger: Optional[logging.Logger] = None,
        memory_saver: Optional[BaseCheckpointSaver] = None,
        subagent_manager: Optional[SubagentManager] = None,
        vector_db_clients: Optional[Dict[str, BaseVectorDBClient]] = None,
        is_master_agent: bool = False,
    ):
        # Read and validate config
        instance = cls(
            config,
            agent_name,
            llm,
            llm_tools_map,
            logger,
            memory_saver,
            subagent_manager,
            vector_db_clients,
            is_master_agent,
        )

        if instance.logger and instance.vector_db_clients:
            client_list = ", ".join(instance.vector_db_clients.keys())
            instance.logger.info(
                f"Agent '{agent_name}' has access to the following vector database clients: {client_list}"
            )
        if instance.use_memory:
            if instance.logger:
                instance.logger.info(
                    f"Memory-aware conversation mode is enabled for the {agent_name} agent"
                )
            if instance.long_term_memory_enabled:
                try:
                    await instance.ainit_long_memory()
                except Exception as e:
                    raise GraphBuildException(
                        f"Failed to initialize long-term memory: {e}"
                    ) from e
            else:
                if instance.logger:
                    instance.logger.info("Long-term memory is disabled")
        else:
            if instance.logger:
                instance.logger.info(
                    f"Memory-aware conversation mode is disabled for the {agent_name} agent"
                )
        try:
            instance.graph = instance.build()
            if instance.logger:
                instance.logger.info("Agent graph built successfully!")
        except Exception as e:
            raise GraphBuildException(f"Failed to build the workflow graph: {e}") from e

        return instance

    async def ainit_long_memory(self):
        """Initialize the long-term memory component"""
        # Get long-term memory configuration
        memory_config = (
            self.config.get("agents", {})
            .get(self.agent_name, {})
            .get("long-term-memory", {})
        )

        # Get configuration values
        collection_name = memory_config.get("collection_name", "memories")
        host = memory_config.get("host", "qdrant")
        port = int(memory_config.get("port", 6333))
        llm_model = memory_config.get("llm_model", "gpt-4.1")
        llm_temperature = float(memory_config.get("llm_temperature", 0.2))
        embedding_model = memory_config.get("embedding_model", "text-embedding-3-large")

        # Initialize long-term memory component
        self.long_term_memory = await AsyncMem0Memory(
            collection_name=collection_name,
            host=host,
            port=port,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            embedding_model=embedding_model,
            logger=self.logger,
        ).aconfigure()

        if self.logger:
            self.logger.info(
                f"Long-term memory initialized with Qdant collection: {collection_name}, Fact extraction model: {llm_model}"
            )

    async def call_model(
        self,
        node_name: str,
        messages_list: List[BaseMessage],
        max_tool_response_tokens: int = 10000,
        trim_tool_responses: bool = False,
        output_model: Optional[Type[BaseModel]] = None,
        as_string: bool = False,
        **llm_kwargs,
    ) -> Any:
        """
        Makes an asynchronous LLM call with a list of messages as input.
        If output_model is provided, returns s,tructured output.
        If as_string is True (and output_model is None), parses output as string.
        """
        # Get the right LLM first for a specific-node
        llm = self.llm_map.get(node_name)
        if not llm:
            raise ModelCallException(
                f"Failed to find the right LLM instance for node: {node_name}"
            )

        # Trim messages to prevent exceeding token limits
        trimmed_messages = safe_trim_llm_message_list(
            messages_list=messages_list,
            llm=llm,
            trim_tool_responses=trim_tool_responses,
            max_tokens_per_tool_msg=max_tool_response_tokens,
            logger=self.logger,
        )

        if self.logger:
            self.logger.debug(
                f"Original message count: {len(messages_list)}, Trimmed count: {len(trimmed_messages)}"
            )

        # Structured output
        try:
            if output_model is not None:
                llm_with_structured = llm.with_structured_output(  # type: ignore
                    output_model, method="function_calling"
                )
                response = await llm_with_structured.ainvoke(
                    trimmed_messages, **llm_kwargs
                )
                return response

            # Unstructured output
            response = await llm.ainvoke(trimmed_messages, **llm_kwargs)
        except Exception as e:
            raise ModelCallException(
                f"Error encountered during model call in non stream-mode: {e}"
            ) from e

        # If the LLM returns tool_calls (as it would via e.g. OpenAI), return as-is.
        # NOTE: Calls will then be processed automatically by the tool_calls node (see self.tool_node)
        # however it falls to the user's node function logic to include the right messages to their next node model call
        # function (containing the tool responses).
        if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
            response.content = "_tool_calls_request_"
            return response
        if as_string:
            response = await StrOutputParser().ainvoke(response)
        return response

    async def call_model_stream(
        self,
        node_name: str,
        messages_list: List[BaseMessage],
        writer: StreamWriter,
        stream_event_key: str,
        stream_structured_output_field: Optional[str] = None,
        output_model: Optional[Type[BaseModel]] = None,
        max_tool_response_tokens: int = 10000,
        trim_tool_responses: bool = False,
        **llm_kwargs,
    ) -> Any:
        """
        Streams unstructured/structured data from an LLM token by token via writer({stream_event_key: ...}).
        This streamed data can be parsed (by the FastAPI's endpoint function) and converted to SSE events.
        - If output_model is provided, parses the full output as structured data and
          emits only the specified stream_structured_output_field via writer({stream_event_key: stream_structured_output_field's  value}) at the end.
        - If output_model is not provided, emits the full text buffer at the end via writer({stream_event_key: buffer}).
        - Returns the final output (structured or unstructured).
        """
        # Get the right LLM first for a specific-node
        llm = self.llm_map.get(node_name)
        if not llm:
            raise ModelCallException(
                f"Failed to find the right LLM instance for node: {node_name}"
            )

        if output_model and not stream_structured_output_field:
            raise ValueError(
                f"Missing a 'stream_structured_output_field' when calling to 'call_model_stream' with a structured output model..."
            )

        trimmed_messages = safe_trim_llm_message_list(
            messages_list=messages_list,
            llm=llm,
            trim_tool_responses=trim_tool_responses,
            max_tokens_per_tool_msg=max_tool_response_tokens,
            logger=self.logger,
        )
        if self.logger:
            self.logger.debug(
                f"Original message count: {len(messages_list)}, Trimmed count: {len(trimmed_messages)}"
            )

        buffer = ""
        tool_calls = []
        gathered = None
        streaming_allowed = True

        # Structured output: get the full response, stream a field, then return the full structured model
        try:
            if output_model is not None:
                structured_response = await llm.with_structured_output(  # type: ignore
                    output_model, method="function_calling"
                ).ainvoke(trimmed_messages, **llm_kwargs)
                field_value = getattr(
                    structured_response, str(stream_structured_output_field), None
                )
                if field_value is not None:
                    writer({stream_event_key: field_value})
                return structured_response
            # Unstructured output: stream tokens, then return the string buffer at the end
            else:
                async for chunk in llm.astream(trimmed_messages, **llm_kwargs):
                    delta = getattr(chunk, "content", None)

                    # Accumulate chunks to merge tool call fragments
                    if gathered is None:
                        gathered = chunk
                    else:
                        gathered = gathered + chunk

                    # Stop streaming as soon as any kind of tool call is detected
                    if getattr(chunk, "tool_calls", None) or getattr(
                        chunk, "tool_call_chunk", None
                    ):
                        streaming_allowed = False

                    # Stream text only if there no tool calls...
                    if streaming_allowed:
                        delta = getattr(chunk, "content", None) or ""
                        buffer += delta
                        writer({stream_event_key: delta})

            # After streaming ends, extract final tool_calls
            tool_calls = getattr(gathered, "tool_calls", []) if gathered else []

            if tool_calls:
                return AIMessage(content="_tool_calls_request_", tool_calls=tool_calls)
            else:
                return AIMessage(content=buffer)
        except Exception as e:
            raise ModelCallException(
                f"Error encountered during model call in stream-mode {e}"
            ) from e

    def _build_forced_tool_call_kwargs(self, fn_name: str) -> Dict:
        """Helper function that returns the kwargs dictionary for a forced tool call of an OpenAI model"""
        tool_choice_kwargs = {
            "tool_choice": {
                "type": "function",
                "function": {"name": fn_name},
            }
        }
        return tool_choice_kwargs

    def build(self):
        """Build and compile the RAG workflow graph from config."""

        # Find the agent configuration
        agents_config = self.config.get("agents", {})
        agent_config = agents_config[self.agent_name]
        if self.logger:
            self.logger.info(f"Building agent: {self.agent_name}")

        # Determine if the selected agent is a "master" by checking for registered subagents
        is_master_agent = hasattr(self, "subagent_manager") and bool(
            getattr(self.subagent_manager, "subagents", {})
        )
        # This will be passed to each node function as a way to pass callable functions to a node (e.g. subagents as "callable functions")
        callable_functions: Dict[str, Any] = {}

        def cast_arg_value(value, type_str):
            """Helper to cast a passed argument to the right type based on the agent's config"""
            if type_str is None:
                return value
            if type_str == "int":
                return int(value)
            if type_str == "float":
                return float(value)
            if type_str == "bool":
                # Accepts "true"/"false" (case-insensitive) or Python bools
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "yes")
            if type_str == "str":
                return str(value)
            if type_str == "dict":
                if isinstance(value, dict):
                    return value
                # Try parsing JSON string
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse dict from string: {e}")
                raise ValueError(
                    f"Cannot cast value of type {type(value).__name__} to dict"
                )
            raise ValueError(f"Unsupported type for casting: {type_str}")

        # Helpers for dynamic imports
        def import_from_path(path: str):
            """Dynamically import a function or class from a string path."""
            module_path, attr = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr)

        def load_prompt_yaml(prompt_path: str):
            with open(prompt_path, "r") as f:
                return yaml.safe_load(f)

        if self.logger:
            self.logger.info("Building graph from config...")

        # --- Get subagent descriptions if available ---
        if self.subagent_manager:
            subagent_descriptions = (
                self.subagent_manager.subagent_descriptions
            )  # textual description of an agent for prompt injection
            subagent_modes = (
                self.subagent_manager.subagent_modes
            )  # e.g. "callable"|"node"
        else:
            subagent_descriptions = {}
            subagent_modes = {}

        # ---  Load the state dict class from the config configuration ---
        state_path = agent_config.get("settings", {}).get("state")
        state_class = import_from_path(state_path)
        workflow = StateGraph(state_class)

        # --- Load and cache all prompt YAMLs ---
        prompt_yamls = {}
        for node_name, node_cfg in agent_config["nodes"].items():
            prompt_path = node_cfg.get("prompt")
            if prompt_path:
                prompt_yamls[node_name] = load_prompt_yaml(prompt_path)

        # --- Find tool calling nodes ---
        tool_using_nodes = {
            node_name
            for node_name, node_cfg in agent_config["nodes"].items()
            if node_cfg.get("tool_usage", False)
        }

        # --- Add Nodes ---

        # --- Handle multi-agent architecture and add compiled sub-agent graphs as nodes ---
        if is_master_agent and self.subagent_manager:
            for subagent_name in self.subagent_manager.list_agents():
                subagent_graph = self.subagent_manager.get_compiled_graph(subagent_name)
                if subagent_graph:
                    create_as = subagent_modes.get(subagent_name, "node")

                    # Create subagent as a node that we can route to from the parent graph (e.g. via "Command")
                    if create_as == "node":
                        workflow.add_node(subagent_name, subagent_graph)
                        if self.logger:
                            self.logger.info(
                                f"Added subagent '{subagent_name}' as a parent graph node"
                            )
                    # Create subagent as a callable function that can be invoked via the standard Langgraph methods (e.g. invoke, ainvoke etc.)
                    elif create_as == "callable_function":
                        callable_functions[subagent_name] = subagent_graph
                        if self.logger:
                            self.logger.info(
                                f"Added subagent '{subagent_name}' as a callable function."
                            )
                            self.logger.debug(
                                "You can call this function in any node function of the parent graph by accessing the 'callable_functions' map from it's passed arguments"
                            )
                    else:
                        self.logger

        # --- Add the configured nodes for current agent ---
        for node_name, node_cfg in agent_config["nodes"].items():
            func = import_from_path(node_cfg["function"])
            prompt_yaml = prompt_yamls.get(node_name, None)

            # If the node has a custom tool_node configured, use that instead of the default ToolNode
            tools_node = ""
            use_default_tool_node = True
            if node_cfg.get("tool_usage", False) and node_cfg.get("tool_node_custom"):
                # Get the custom tool node's name mapping
                custom_node = node_cfg["tool_node_custom"]
                if isinstance(custom_node, list):
                    custom_node = custom_node[0]

                # Only use it if it exists in the agent config nodes...
                if custom_node in agent_config["nodes"]:
                    tools_node = custom_node
                    use_default_tool_node = False  # do not create the default ToolNode, we have a custom one
                else:
                    if self.logger:
                        self.logger.warning(
                            f"Custom tool node '{custom_node}' for '{node_name}' is not defined in agent config nodes!"
                        )

                # Log for transparency
                if self.logger:
                    self.logger.debug(
                        f"Using custom tool execution node with name '{custom_node}' for node '{node_name}' instead of a LangGraph ToolNode..."
                    )
            elif node_name in tool_using_nodes:
                tools_node = node_name + "_tools"

            # --- Propagate shared dependencies to all node functions as arguments ---
            # NOTE: We use this to propagate resources from the graph to the node functions

            # If this node has a custom tool execution node, copy this node's tools_map
            # to the custom tool node so it has the same list of tools available. This makes
            # it easy to access the right tools for execution inside the custom tool node.
            tools_map_value = None
            if (
                node_cfg.get("tool_usage", False)
                and node_cfg.get("tool_node_custom")
                and tools_node in agent_config["nodes"]
            ):
                parent_tools_list = self.tools_map_per_node.get(node_name, [])
                self.tools_map_per_node[tools_node] = parent_tools_list
                tools_map_value = parent_tools_list
            elif node_name in tool_using_nodes:  # In case we use default node...
                tools_map_value = self.tools_map_per_node.get(node_name)
            elif (
                node_name in self.tools_map_per_node.keys()
            ):  # For a custom tool node...
                tools_map_value = self.tools_map_per_node.get(node_name)

            # Load any extra configurations for the agent from the 'settings' key
            agent_custom_settings = agent_config.get("settings", {}).get("custom", {})

            kwargs = {
                "prompt_yaml": prompt_yaml,
                "prompt_builder": self.prompt_builder,
                "logger": self.logger,
                "forced_tool_call_args_fn": getattr(
                    self, "_build_forced_tool_call_kwargs", None
                ),
                "long_memory_store_mem0": (
                    self.long_term_memory if self.use_memory else None
                ),
                "model_call_fn": functools.partial(self.call_model, node_name),
                "stream_model_call_fn": functools.partial(
                    self.call_model_stream, node_name
                ),
                "vector_db_clients": self.vector_db_clients,  # Pass all clients to databases available to the agent to the node functions...
                "tools_map": tools_map_value,
                "tools_node_name": tools_node,
                "callable_functions": callable_functions,
                "custom_settings": agent_custom_settings,
            }

            # Add the subagent descriptions only to master agent!
            if is_master_agent:
                kwargs["subagents"] = subagent_descriptions

            # --- Handle extra_args if those are provided for a node function ---
            # NOTE: if value is "self", use self.<arg>, else use the value directly
            for arg, arg_cfg in node_cfg.get("extra_args", {}).items():
                # Support both older config (only value) and newer one(dict format)...
                if isinstance(arg_cfg, dict):
                    value = arg_cfg.get("value")
                    type_str = arg_cfg.get("type")
                else:
                    value = arg_cfg
                    type_str = None
                if value == "self":
                    # Check if the self attribute exists in the class...
                    if hasattr(self, arg):
                        attr = getattr(self, arg)
                        if attr is None:
                            raise ValueError(
                                f"Attribute '{arg}' on self is None for node '{node_name}'"
                            )

                        # --- Special handling: if the attr is a dict of per-node values, extract this node's cfg ---
                        if isinstance(attr, dict) and node_name in attr:
                            kwargs[arg] = attr[node_name]
                        else:
                            kwargs[arg] = attr
                    else:
                        raise ValueError(
                            f"Unknown attribute '{arg}' on self for node '{node_name}'"
                        )
                else:
                    kwargs[arg] = cast_arg_value(value, type_str)

            # --- Inspect signature and only keep the expected function args ---
            sig = inspect.signature(func)
            accepted_args = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}

            # --- Build the node ---
            node_fn = functools.partial(func, **filtered_kwargs)
            workflow.add_node(node_name, node_fn)

            # --- If this node uses tools, create a ToolNode for it and link with edges ---
            if node_name in tool_using_nodes and use_default_tool_node:
                tool_dict = self.tools_map_per_node.get(node_name, {})
                if tool_dict:
                    tool_node_name = f"{node_name}_tools"

                    # Override default messages key for tool node in case one other than the standard (e.g. "messages") is provided
                    tool_message_key = node_cfg.get("tool_message_key", "messages")
                    self.tool_nodes_per_node[node_name] = ToolNode(
                        tools=list(
                            tool_dict.values()  # type: ignore
                        ),  # List of Base tools without name/BaseTool format...
                        handle_tool_errors=True,
                        messages_key=tool_message_key,
                    )
                    workflow.add_node(
                        tool_node_name, self.tool_nodes_per_node[node_name]
                    )

                    # Add edge from tool node back to original node
                    workflow.add_edge(tool_node_name, node_name)

        # --- Define the agent's entry point ---
        entry_point = agent_config["entry_point"]

        if not isinstance(entry_point, str):
            raise GraphBuildException(
                f"The field 'entry_point' in the `agent_config` file should be a single string but is: {type(entry_point)}"
            )
        workflow.set_entry_point(cast(str, entry_point))

        # --- Compile the graph ---
        graph = workflow.compile(
            checkpointer=self.memory if self.use_memory else None,
            store=self.in_memory_store,
        )
        return graph

    async def ask(
        self,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """
        Run the workflow graph with a given initial state.

        Args:
            initial_state (dict): The initial state for the graph.
            config (dict, optional): Config for the graph run (e.g., recursion_limit, thread_id, etc).
            stream: Enables output streaming mode (this will produce SSE events)
        Returns:
            The result of the graph run.
        """
        # NOTE: Should never happen! Just here for safety
        if not self.graph:
            raise GraphBuildException(
                "The agent seems to not have a compiled graph available for execution"
            )

        # Set default config with recursion_limit if not provided
        if config is None:
            config = {"recursion_limit": 100}
        elif "recursion_limit" not in config:
            config = {**config, "recursion_limit": 100}

        # Handle memory mode and set the conversation thread
        if self.use_memory:
            thread_id = config.get("thread_id")
            user_id = config.get("user_id", "default_user")
            if not thread_id:
                raise ValueError(
                    "Memory mode is enabled, but no 'thread_id' provided in config."
                )

            config = {
                "configurable": {"thread_id": thread_id, "user_id": user_id},
                **config,
            }

            graph_to_run = self.graph.with_config(checkpointer=self.memory)
        else:
            graph_to_run = self.graph

        if stream:
            # Return the async generator for streaming
            return graph_to_run.astream(
                initial_state, config=config, stream_mode="custom"
            )
        else:
            # Return the final result (non-streaming)
            result = await graph_to_run.ainvoke(initial_state, config=config)  # type: ignore
            return result
