"""
Agent builder function for dependency injection
"""

import os
import yaml
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from tavily import TavilyClient
from src.agent_graph.graph import AgentGraph
from backend_api.factories.language_models.model_factory import LangchainModelFactory
from src.resources.agent_manager.manager import SubagentManager
from backend_api.helpers.helpers import log_agent_configurations
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.resources.vector_database_clients.base import BaseVectorDBClient
from src.resources.external_modules.tavily_manager import TavilyWebSearchManager
from src.resources.tools.web_search_tavily import set_tavily_manager_for_tools
from backend_api.factories.databases.neo4j import Neo4jClientFactory
from backend_api.factories.databases.qdrant import QdrantClientFactory
from urllib.parse import quote_plus
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def initialize_tavily_web_search(
    config_data: Dict, logger: Optional[logging.Logger] = None
):
    """Initializer for the Tavily web search manager"""
    global _tavily_web_search_manager

    def find_tavily_web_search_config(config: dict) -> dict | None:
        """
        Search all agents in the config for a web_search of type 'tavily'.
        Returns the first matching agent's web_search config or None if not found.
        """
        agents = config.get("agents", {})
        for agent_name, agent_cfg in agents.items():
            web_search_cfg = agent_cfg.get("settings", {}).get("web_search", {})
            if "tavily" in web_search_cfg:
                return web_search_cfg["tavily"]
        return None

    tavily_settings = find_tavily_web_search_config(config_data)
    if tavily_settings:
        if logger:
            logger.info(
                "One or more agents are configured with Tavily web-search. Setting up the Tavily web search client...."
            )
    else:
        return None

    # Initialize the Tavily Web Search manager
    tavily_client = TavilyClient()
    _tavily_web_search_manager = TavilyWebSearchManager()
    TavilyWebSearchManager.set_tavily_client(tavily_client)

    # Initialize summarization model
    model_cfg = tavily_settings.get("web_content_summarization_model")
    if not model_cfg:
        raise RuntimeError(
            "Missing 'web_content_summarization_model' in Tavily web search settings for the agent."
        )

    provider = model_cfg.get("provider")
    if not provider:
        raise RuntimeError(
            "Missing 'provider' in 'web_content_summarization_model' section."
        )
    model_name = model_cfg.get("model")
    if not model_name:
        raise RuntimeError(
            "Missing 'model' in 'web_content_summarization_model' section."
        )
    model_args = model_cfg.get("model_args", {})

    # Look up API key from global model-secrets
    model_secrets = config_data.get("global-settings", {}).get("model-secrets", {})
    api_key_env_var = model_secrets.get(provider, {}).get("api_key_env_var")
    if not api_key_env_var:
        raise RuntimeError(
            f"Missing API key env variable configuration for provider '{provider}' when setting up the Tavily web-search manager."
        )

    summarization_model = TavilyWebSearchManager._get_llm_for_provider(
        provider=provider,
        model_name=model_name,
        model_args=model_args,
        model_secrets=model_secrets,
    )
    web_summarization_prompt_path = tavily_settings.get(
        "web_content_summarization_prompt"
    )
    if not web_summarization_prompt_path:
        raise RuntimeError("No web_content_summarization_prompt specified in config")

    # Set the summarization model for the web-search tool
    TavilyWebSearchManager.set_summarization_model(
        summarization_model, web_summarization_prompt_path
    )

    # Make the client manager singleton available to tools
    set_tavily_manager_for_tools(_tavily_web_search_manager)

    return _tavily_web_search_manager


async def initialize_postgres_short_memory():
    """
    Initialize the AsyncConnectionPool and AsyncPostgresSaver for short-term memory.
    """
    global _short_memory_pool, _short_memory

    db_user = quote_plus(os.getenv("POSTGRES_USER", ""))
    db_password = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
    db_name = quote_plus(os.getenv("POSTGRES_DB", "short-memory"))
    db_host = os.getenv("POSTGRES_HOST", "")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    db_uri = f"postgresql://{db_user}:{db_password}@/{db_name}?host={db_host}&port={db_port}&sslmode=disable"

    _short_memory_pool = AsyncConnectionPool(
        db_uri, timeout=60, kwargs={"autocommit": True, "prepare_threshold": None}
    )
    _short_memory = AsyncPostgresSaver(_short_memory_pool)  # type: ignore
    await _short_memory.setup()

    return _short_memory, _short_memory_pool


def initialize_db_client(config_data: Dict, logger, client_name: str = "") -> dict:
    """Client initiliazer for interaction with different databases"""
    global_settings = config_data.get("global-settings", {})
    model_secrets = global_settings.get("model-secrets", {})
    databases = global_settings.get("databases", {})

    if not databases:
        raise ValueError("No databases configured.")

    if not client_name:
        raise ValueError(
            "No database client name specified, failed to look for one in config."
        )

    if client_name not in databases:
        raise ValueError(f"Database client '{client_name}' not found in config.")

    # NOTE: On extension of the framework with more clients we should add those here!
    factories = {"neo4j": Neo4jClientFactory(), "qdrant": QdrantClientFactory()}

    db_config = databases[client_name]
    db_type = db_config.get("type")
    if not db_type:
        raise ValueError(
            f"Database client '{client_name}' missing required 'type' field."
        )

    if db_type not in factories:
        raise ValueError(f"Unsupported database type: {db_type}")

    factory = factories[db_type]

    try:
        return factory.create(
            db_config, logger, model_secrets=model_secrets, client_name_key=client_name
        )
    except Exception as e:
        raise


def initialize_all_vector_db_clients(
    config_data: Dict,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    _all_db_clients = {}

    databases_config = config_data.get("global-settings", {}).get("databases", {})
    if not databases_config:
        if logger:
            logger.debug(
                "No vector database client configurations found in 'global-settings.databases'..."
            )
        return _all_db_clients

    if logger:
        logger.info("Creating clients for the vector database configurations found...")

    for client_name in databases_config.keys():
        result = initialize_db_client(
            config_data=config_data,
            logger=logger,
            client_name=client_name,
        )
        _all_db_clients[client_name] = result["client"]

    if logger:
        if _all_db_clients:
            logger.debug(
                "The following vector database clients have been succesfully initialized:\n - "
                + "\n - ".join(_all_db_clients.keys())
            )
        else:
            logger.debug(
                "No vector database clients were initialized for the application..."
            )

    return _all_db_clients


async def initialize_factory(config_data: Dict, logger: logging.Logger):
    """Initializer function for the LLM factory"""
    # Initialize the factory
    factory = LangchainModelFactory(config_data, logger=logger)
    await factory.initialize()
    return factory


async def initialize_agent(
    config_data: Dict,
    logger: logging.Logger,
    env_log_level: str,
    master_agent_name: str,
    llm_factory: Optional[LangchainModelFactory] = None,
    memory_saver: Optional[BaseCheckpointSaver] = None,
    agent_name: str = "",
    log_config: bool = False,
    vector_db_clients: Optional[Dict[str, BaseVectorDBClient]] = None,
):
    """Initialization function for the LangGraph agent"""
    if llm_factory is None:
        raise ValueError(
            "llm_factory must be provided and initialized before calling initialize_agent."
        )

    # Load agents config
    agents_config = config_data["agents"]

    # Pick agent to initialize, default to 'master' agent first....
    if not agent_name:
        agent_name = master_agent_name

    agent_config = agents_config[agent_name]
    settings = agent_config.get("settings", {})

    # Pick any subagents for each agent.
    # NOTE: Each agent should only see its own subagents
    subagent_entries = settings.get("subagents", [])

    # Export the models schema dicts from the initialized factory
    factory_schema = llm_factory.export()
    node_map = factory_schema[agent_name]
    llm_per_node = {node: node_map[node]["llm_config"] for node in node_map}
    tools_per_node = {node: node_map[node]["tools_map"] for node in node_map}
    agents_list = list(factory_schema.keys())

    if log_config:
        log_lines = [
            "Initializing agentic application with the following configuration:",
            f" - Langsmith Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}",
            f" - Logging level: {env_log_level}",
            f" - Config file: {os.getenv('AGENT_CONFIG')}",
            f" - Agents: {agents_list}",
            f" - Master agent: '{master_agent_name}'",
        ]

        if logger:
            logger.info("\n".join(log_lines))
            # NOTE: This will log the node-level detailed setup configurations for all configured agents
            log_agent_configurations(logger, factory_schema, agents_config)

    # --- Subagent logic ---
    subagent_manager = None
    if subagent_entries:
        if logger:
            logger.debug(
                f"Initializing sub-agents manager for parent agent: {agent_name}"
            )
        subagent_manager = SubagentManager(logger=logger)
        for subagent_entry in subagent_entries:
            if isinstance(subagent_entry, dict):
                subagent_name = subagent_entry["name"]
                subagent_description = subagent_entry.get(
                    "description", "No description is available for this subagent"
                )

                # Options:
                #   - callable : Passed to node functions (via the ) as a compiled graph
                #                (e.g. subagent.invoke())
                #   - node     : Injected directly as a node of the parent graph
                subagent_as = subagent_entry.get("as", "node")
            else:
                subagent_name = subagent_entry
                subagent_description = ""
                subagent_as = "node"
            # Recursively initialize subagent
            subagent_graph = await initialize_agent(
                config_data=config_data,
                logger=logger,
                master_agent_name=master_agent_name,
                env_log_level=env_log_level,
                llm_factory=llm_factory,
                memory_saver=None,
                agent_name=subagent_name,
                log_config=False,  # No need to recursively log here..We will log all info from the main agent.
                vector_db_clients=vector_db_clients,
            )
            subagent_manager.register_agent(
                subagent_name,
                subagent_graph,
                description=subagent_description,
                create_as=subagent_as,
            )

    # --- Main agent ---
    graph = await AgentGraph.acreate(
        config=config_data,
        agent_name=agent_name,
        llm=llm_per_node,
        llm_tools_map=tools_per_node,
        logger=logger,
        memory_saver=memory_saver,
        subagent_manager=subagent_manager,
        vector_db_clients=vector_db_clients,
        is_master_agent=(agent_name == master_agent_name),
    )
    return graph
