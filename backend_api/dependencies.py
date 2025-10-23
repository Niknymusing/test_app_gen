import os
import asyncio
from typing import Annotated, Optional, Dict, Tuple
from fastapi import Depends
from pathlib import Path
from backend_api.logger import agent_logger, env_log_level
from src.agent_graph.graph import AgentGraph
from src.resources.external_modules.tavily_manager import TavilyWebSearchManager
from backend_api.initializers import (
    initialize_agent,
    initialize_factory,
    initialize_postgres_short_memory,
    initialize_all_vector_db_clients,
    initialize_tavily_web_search,
)
from backend_api.helpers.helpers import (
    is_postgres_short_memory_enabled,
    find_master_agent,
    read_yaml_config,
)
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

_agent_graph = None
_llm_factory = None
_all_db_clients = {}
_short_memory_pool: Optional[AsyncConnectionPool] = None
_short_memory: Optional[AsyncPostgresSaver] = None
_tavily_web_search_manager: Optional[TavilyWebSearchManager] = None
_mcp_clients: Dict[Tuple[str, str], MultiServerMCPClient] = {}


async def initialize_dependencies():
    """
    Initialize global agent dependencies with retry logic.
    """
    global \
        _llm_factory, \
        _agent_graph, \
        _all_db_clients, \
        _short_memory, \
        _short_memory_pool, \
        _tavily_web_search_manager, \
        _mcp_clients

    max_retries = 3
    retry_interval = 10
    last_exception = None
    for attempt in range(max_retries):
        try:
            # --- Read the config file once ---
            config_file_path = Path(os.getenv("AGENT_CONFIG", "agent_config.yaml"))
            config_file_data = read_yaml_config(config_path=config_file_path)

            # --- Initialize the model factory ---
            _llm_factory = await initialize_factory(config_file_data, agent_logger)

            # --- Get any initialized MCP clients from the factory (for logging & cleanup) ---
            _mcp_clients = _llm_factory.get_all_mcp_clients()

            # --- Initialize all vector DB clients ---
            _all_db_clients = initialize_all_vector_db_clients(
                config_data=config_file_data, logger=agent_logger
            )

            # Initialize short-term memory on a PostGreSQL database, if that is enabled
            use_postgres_short_memory = is_postgres_short_memory_enabled(
                config_data=config_file_data, logger=agent_logger
            )
            if use_postgres_short_memory:
                (
                    _short_memory,
                    _short_memory_pool,
                ) = await initialize_postgres_short_memory()  # type: ignore
            else:
                _short_memory = None
                _short_memory_pool = None
                agent_logger.info(
                    "Using a PostgreSQL database to store short-term memory is disabled."
                    " Will default to in-memory saving (if memory-usage is set)..."
                )

            # Initialize a Tavily web-search client if web-search is enabled for any agent
            _tavily_web_search_manager = initialize_tavily_web_search(
                config_file_data, logger=agent_logger
            )

            # Initialize agents

            # Find out which agent is the 'master' agent (the agent called 'master', else the first in config in multi-agent setups)
            master_name, _ = find_master_agent(
                config_data=config_file_data, logger=agent_logger
            )

            _agent_graph = await initialize_agent(
                config_data=config_file_data,
                logger=agent_logger,
                env_log_level=env_log_level,
                llm_factory=_llm_factory,
                memory_saver=_short_memory,
                log_config=True,  # This enables logging for the main agent!
                vector_db_clients=_all_db_clients,
                master_agent_name=master_name,
            )
            if _agent_graph is None:
                raise RuntimeError("Agent graph initialization returned None")

            agent_logger.info("All agent dependencies initialized successfully.")
            return  # Success

        except Exception as e:
            last_exception = e
            agent_logger.warning(
                f"Failed to initialize dependencies on attempt {attempt + 1}/{max_retries}: {e}"
            )
            # Clean up any partially initialized resources before retrying
            await cleanup_dependencies()

            if attempt < max_retries - 1:
                agent_logger.info(f"Retrying in {retry_interval} seconds...")
                await asyncio.sleep(retry_interval)
            else:
                agent_logger.error("Exhausted all retries to initialize dependencies.")
                raise RuntimeError(
                    f"Failed to initialize dependencies after {max_retries} attempts"
                ) from last_exception


async def get_agent_graph() -> AgentGraph:
    global _agent_graph
    if _agent_graph is None:
        agent_logger.error("Agent graph not initialized")
        raise RuntimeError("Agent graph not initialized")
    return _agent_graph


AgentGraphDep = Annotated[AgentGraph, Depends(get_agent_graph)]


async def cleanup_dependencies():
    """
    Clean up and release all global resources.
    """
    global \
        _agent_graph, \
        _llm_factory, \
        _all_db_clients, \
        _short_memory, \
        _short_memory_pool, \
        _tavily_web_search_manager, \
        _mcp_clients
    # Close the connection to the PostGreSQL database that holds the short memory if one is configured...
    if _short_memory_pool is not None:
        await _short_memory_pool.close()
        _short_memory_pool = None

    _agent_graph = None
    _all_db_clients = {}
    _short_memory = None
    _llm_factory = None
    _tavily_web_search_manager = None
    _mcp_clients = {}
    agent_logger.info("Agent dependencies cleaned up")
