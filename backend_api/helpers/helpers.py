"""
FastAPI related helper code
"""

import traceback
from fastapi import HTTPException, status
from pathlib import Path
import yaml
import logging
from typing import NoReturn, Any, Dict, Optional, Tuple


def read_yaml_config(config_path: Path) -> dict:
    """
    Loads the workflow graph configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is not valid YAML or missing required keys.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config file not found: {config_path}")

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file {config_path} does not contain a valid YAML dictionary."
        )

    # Top-level required keys
    required_top_keys = ["global-settings", "agents"]
    for key in required_top_keys:
        if key not in config:
            raise ValueError(
                f"Config file {config_path} is missing required key: '{key}'"
            )

    # Per-agent required keys
    required_agent_keys = ["settings", "nodes", "entry_point"]
    for agent, agent_conf in config["agents"].items():
        for key in required_agent_keys:
            if key not in agent_conf:
                raise ValueError(
                    f"Agent '{agent}' in config file {config_path} is missing required key: '{key}'"
                )

    return config


def raise_internal_error(request, agent_logger, e) -> NoReturn:
    error_message = f"An unexpected error occurred: {str(e)}"
    if agent_logger:
        agent_logger.error(
            f"Error processing query '{request.query}': {error_message}\n{traceback.format_exc()}",
            exc_info=True,
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": error_message,
            "details": "Seems like an internal server error occurred in the agent's pipeline!",
        },
    )


def log_agent_configurations(
    logger: logging.Logger,
    models_schema: Dict[str, Any],
    agents_config: Dict[str, Any],
):
    """
    Logs the configurations of all agents at node-level detail (when on DEBUG log level),
    including subagent information pulled directly from the YAML config.
    """
    for agent_name, node_map in models_schema.items():
        logger.debug(f"{'-' * 90}")
        logger.debug(f"Configurations for the '{agent_name}' agent:")

        # --- Log subagents from YAML config ---
        agent_config = agents_config.get(agent_name, {})
        subagent_entries = agent_config.get("settings", {}).get("subagents", [])

        if subagent_entries:
            logger.debug("  Subagents:")
            for entry in subagent_entries:
                if isinstance(entry, dict):
                    name = entry.get("name", "Unnamed subagent")
                    desc = entry.get("description", "No description")
                    logger.debug(f"    - {name}: {desc}")
                else:
                    logger.debug(f"    - {entry}")
        else:
            logger.debug("  Subagents: None")

        # --- Log nodes (keep this part intact) ---
        logger.debug("  Nodes: ")
        for node, node_info in node_map.items():
            tools = node_info.get("tools_map", "Missing tools map")
            provider = node_info.get("provider", "Missing provider")
            model_name = node_info.get("model_name", "Missing model name")

            if tools and isinstance(tools, dict) and tools.keys():
                tool_str = ", ".join(tools.keys())
            else:
                tool_str = "None"

            logger.debug(
                f"    - Node '{node}': Provider: {provider} | Model: {model_name} | Tools: {tool_str}"
            )

    logger.debug(f"{'-' * 90}")


def find_master_agent(
    config_data: Dict, logger: Optional[logging.Logger] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Identify and return the master agent from a configuration.

    This function looks up agents defined under the "agents" key in the
    given configuration dictionary. If an agent named "master" exists, it
    will be selected as the master agent. Otherwise, the first defined agent
    will be used as the fallback master. The function returns both the
    master agent's name and its configuration.

    Args:
        config_data (Dict): Configuration data (from agent config file).
        logger (Optional[logging.Logger]): Optional logger for debug/info messages.

    Returns:
        Tuple[str, Dict]: A tuple of (master_agent_name, master_agent_config).
        Returns (None, None) if no agents are defined.

    Notes:
        - If multiple agents are defined, only the master agent is considered
          for memory setup or related configurations.
        - The case where no agents exist should never occur under normal usage,
          but is handled defensively.
    """
    agents = config_data.get("agents", {})

    if not agents:
        raise ValueError("No agents defined in configuration.")

    # Step 1: Pick 'master' if available, else first agent
    if "master" in agents:
        master_name = "master"
        master_config = agents[master_name]
        if logger:
            logger.info(
                "Using the 'master' agent to check memory type/usage ('memory_type': [PostgreSQL] | 'use_memory': [true, false]) configurations..."
            )
    else:
        master_name = next(iter(agents))
        master_config = agents[master_name]
        if logger:
            logger.info(
                f"Using the '{master_name}' agent as 'master' agent to check memory type/usage ('memory_type': [PostgreSQL] | 'use_memory': [true, false]) configurations..."
            )

    if len(agents) > 1 and logger:
        logger.debug(
            "Multiple agents defined in the configuration. "
            "If using a master-subagent setup, memory will only be set for the master agent."
        )

    assert isinstance(master_name, str)

    return master_name, master_config


def is_postgres_short_memory_enabled(
    config_data: Dict, logger: Optional[logging.Logger] = None
) -> bool:
    """
    Check if the 'master' agent (if exists) or the first available agent
    has memory enabled and the memory type is 'postgres'.

    Args:
        config_data (Dict): Configuration data (from agent config file).
        logger (Optional[logging.Logger]): Optional logger for debug/info messages.

    Returns True only if:
      - use_memory is True (or not specified), and
      - memory_type is 'postgres'.
    """
    # Currently, for simplicity we only support memory setup for one/or the 'master' agent in a multi-agent setup
    _, master_config = find_master_agent(config_data, logger)

    if not master_config:
        return False  # No agents defined: SHOULD NEVER HAPPEN at this point!

    # Step 2: Get settings
    agent_settings = master_config.get("settings", {})

    use_memory = agent_settings.get("use_memory", False)
    memory_type = agent_settings.get("memory_type", "").lower()

    return use_memory and memory_type == "postgres"
