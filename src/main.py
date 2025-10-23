"""
MEU Framework Coder Agent - Main Entry Point

This is the main entry point for the MEU (Model-Execute-Update) framework coder agent.
It initializes the agent graph, sets up communication protocols, and starts the agent.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

from meu_agent import MEUCoderAgentGraph
from resources.tools.a2a import discover_agents
from resources.tools.mcp import get_resources
from resources.tools.meu import update_workspace_state

# Create logs directory if it doesn't exist
workspace_path = os.getenv("WORKSPACE_PATH", "/tmp/test-workspace")
logs_dir = os.path.join(workspace_path, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, 'meu_coder_agent.log'), mode='a')
    ]
)

logger = logging.getLogger(__name__)


async def initialize_agent():
    """Initialize the MEU coder agent with all necessary components."""

    try:
        logger.info("Initializing MEU Coder Agent...")

        # Initialize workspace
        workspace_path = os.getenv("WORKSPACE_PATH", "/tmp/test-workspace")
        agent_id = os.getenv("AGENT_ID", "meu-coder-001")

        # Create logs directory if it doesn't exist
        os.makedirs(f"{workspace_path}/logs", exist_ok=True)

        # Initialize MEU agent graph
        agent_graph = MEUCoderAgentGraph(
            workspace_path=workspace_path,
            agent_id=agent_id
        )

        # Test A2A connectivity
        logger.info("Testing A2A connectivity...")
        agents_result = discover_agents()
        if agents_result["success"]:
            logger.info(f"Discovered {len(agents_result['agents'])} agents in A2A network")
        else:
            logger.warning(f"A2A discovery failed: {agents_result['error']}")

        # Test MCP connectivity
        logger.info("Testing MCP connectivity...")
        mcp_result = get_resources()
        if mcp_result["success"]:
            logger.info(f"MCP connected with {mcp_result['total_count']} resources available")
        else:
            logger.warning(f"MCP connection failed: {mcp_result['error']}")

        # Initialize workspace state
        initial_state = {
            "operation": "agent_initialization",
            "data": {
                "agent_id": agent_id,
                "workspace_path": workspace_path,
                "a2a_connectivity": agents_result["success"],
                "mcp_connectivity": mcp_result["success"],
                "initialized_at": "timestamp"
            }
        }

        workspace_result = update_workspace_state(initial_state)
        if workspace_result["success"]:
            logger.info("Workspace state initialized successfully")
        else:
            logger.warning(f"Workspace initialization warning: {workspace_result['error']}")

        logger.info("MEU Coder Agent initialized successfully")
        return agent_graph

    except Exception as e:
        logger.error(f"Failed to initialize MEU Coder Agent: {e}")
        raise


async def start_agent_server():
    """Start the agent server with FastAPI backend."""

    try:
        # Import FastAPI components
        from backend.api import create_app
        import uvicorn

        # Initialize agent
        agent_graph = await initialize_agent()

        # Create FastAPI app with agent
        app = create_app(agent_graph)

        # Get server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))

        logger.info(f"Starting MEU Coder Agent server on {host}:{port}")

        # Start the server
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()

    except Exception as e:
        logger.error(f"Failed to start agent server: {e}")
        raise


async def run_standalone_mode():
    """Run agent in standalone mode for testing."""

    try:
        logger.info("Starting MEU Coder Agent in standalone mode")

        # Initialize agent
        agent_graph = await initialize_agent()

        # Test workflow with a sample request
        test_request = "Implement a simple calculator function with add, subtract, multiply, and divide operations"
        workspace_path = os.getenv("WORKSPACE_PATH", "/tmp/test-workspace")

        logger.info("Testing MEU workflow with sample request...")

        # Create initial state
        initial_state = {
            "user_request": test_request,
            "workspace_path": workspace_path,
            "mode": "standalone_test"
        }

        # Execute the workflow
        result = await agent_graph.execute(initial_state)

        logger.info("MEU workflow test completed")
        logger.info(f"Final status: {result.get('status', 'unknown')}")

        if result.get("evaluation_summary"):
            eval_summary = result["evaluation_summary"]
            logger.info(f"Overall score: {eval_summary.get('success_rate', 0.0):.2f}")
            logger.info(f"Final status: {eval_summary.get('final_status', 'unknown')}")

        return result

    except Exception as e:
        logger.error(f"Standalone mode execution failed: {e}")
        raise


def main():
    """Main entry point for the MEU coder agent."""

    try:
        # Determine run mode
        mode = os.getenv("RUN_MODE", "server")  # "server" or "standalone"

        if mode == "standalone":
            # Run in standalone mode
            result = asyncio.run(run_standalone_mode())
            print(f"Standalone execution completed with status: {result.get('status', 'unknown')}")
        else:
            # Run as server
            asyncio.run(start_agent_server())

    except KeyboardInterrupt:
        logger.info("Agent shutdown requested by user")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()