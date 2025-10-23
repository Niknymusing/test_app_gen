"""
A2A (Agent-to-Agent) Communication Tools

This module implements A2A protocol communication for MEU framework agents,
enabling distributed collaboration and task delegation between agents.
"""

import json
import logging
import os
import requests
import jwt
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class A2AProtocolClient:
    """Client for A2A protocol communication"""

    def __init__(self, discovery_uri: str = None, agent_id: str = None, jwt_secret: str = None):
        self.discovery_uri = discovery_uri or os.getenv("A2A_DISCOVERY_URI", "http://a2a-discovery:8101")
        self.agent_id = agent_id or os.getenv("A2A_AGENT_ID", "meu-coder-001")
        self.jwt_secret = jwt_secret or os.getenv("A2A_JWT_SECRET", "default-secret")
        self.session = requests.Session()
        self._setup_authentication()

    def _setup_authentication(self):
        """Setup JWT authentication for A2A communication"""
        try:
            payload = {
                "agent_id": self.agent_id,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=1)
            }
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            self.session.headers.update({
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            })
        except Exception as e:
            logger.warning(f"Failed to setup A2A authentication: {e}")

    def discover_agents(self) -> Dict[str, Any]:
        """Discover available agents in the network"""
        try:
            response = self.session.get(f"{self.discovery_uri}/agents.json", timeout=10)
            if response.status_code == 200:
                return {
                    "success": True,
                    "agents": response.json().get("agents", []),
                    "discovery_uri": self.discovery_uri
                }
            else:
                return {
                    "success": False,
                    "error": f"Discovery service returned status {response.status_code}",
                    "discovery_uri": self.discovery_uri
                }
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return {
                "success": False,
                "error": str(e),
                "discovery_uri": self.discovery_uri
            }

    def send_message(self, target_agent_id: str, message_type: str, payload: Dict[str, Any],
                    timeout: int = 30) -> Dict[str, Any]:
        """Send a message to another agent"""
        try:
            # First discover the target agent
            agents_response = self.discover_agents()
            if not agents_response["success"]:
                return {
                    "success": False,
                    "error": "Failed to discover agents",
                    "target_agent": target_agent_id
                }

            # Find target agent endpoint
            target_endpoint = None
            for agent in agents_response["agents"]:
                if agent["agent_id"] == target_agent_id:
                    target_endpoint = agent["endpoint"]
                    break

            if not target_endpoint:
                return {
                    "success": False,
                    "error": f"Agent {target_agent_id} not found in discovery",
                    "target_agent": target_agent_id
                }

            # Prepare A2A message
            a2a_message = {
                "message_id": f"{self.agent_id}-{datetime.utcnow().isoformat()}",
                "sender_id": self.agent_id,
                "target_id": target_agent_id,
                "message_type": message_type,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload
            }

            # Send message
            response = self.session.post(
                f"{target_endpoint}/api/v1/a2a/message",
                json=a2a_message,
                timeout=timeout
            )

            if response.status_code in [200, 201, 202]:
                return {
                    "success": True,
                    "response": response.json(),
                    "target_agent": target_agent_id,
                    "message_type": message_type,
                    "message_id": a2a_message["message_id"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Target agent returned status {response.status_code}",
                    "response_text": response.text,
                    "target_agent": target_agent_id
                }

        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_agent": target_agent_id
            }


# Global A2A client instance
_a2a_client = A2AProtocolClient()


def discover_agents() -> Dict[str, Any]:
    """
    Discover available agents in the A2A network

    Returns:
        Dict containing list of discovered agents and their capabilities
    """
    try:
        result = _a2a_client.discover_agents()

        if result["success"]:
            # Enhance with capability analysis
            agents = result["agents"]
            capability_summary = {}

            for agent in agents:
                capabilities = agent.get("capabilities", [])
                for capability in capabilities:
                    if capability not in capability_summary:
                        capability_summary[capability] = []
                    capability_summary[capability].append(agent["agent_id"])

            result["capability_summary"] = capability_summary
            result["total_agents"] = len(agents)

        return result

    except Exception as e:
        logger.error(f"Error in discover_agents: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def send_message(target_agent_id: str, message_type: str, content: str,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Send a message to another agent using A2A protocol

    Args:
        target_agent_id: ID of the target agent
        message_type: Type of message (e.g., "task_request", "collaboration", "query")
        content: Message content
        context: Additional context data

    Returns:
        Dict containing response from target agent
    """
    try:
        payload = {
            "content": content,
            "context": context or {},
            "sender_capabilities": [
                "meu_framework",
                "code_implementation",
                "test_automation",
                "claude_code_integration"
            ]
        }

        return _a2a_client.send_message(target_agent_id, message_type, payload)

    except Exception as e:
        logger.error(f"Error sending message to {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def request_collaboration(target_agent_id: str, task_description: str,
                         required_capabilities: List[str],
                         meu_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Request collaboration from another agent for a specific task

    Args:
        target_agent_id: ID of the target agent to collaborate with
        task_description: Description of the task requiring collaboration
        required_capabilities: List of capabilities needed for the task
        meu_context: MEU framework context (triplet info, workspace state, etc.)

    Returns:
        Dict containing collaboration response
    """
    try:
        collaboration_payload = {
            "task_description": task_description,
            "required_capabilities": required_capabilities,
            "meu_context": meu_context or {},
            "collaboration_type": "meu_framework",
            "expected_deliverables": [
                "task_completion_status",
                "results_data",
                "any_generated_artifacts"
            ]
        }

        result = _a2a_client.send_message(
            target_agent_id,
            "collaboration_request",
            collaboration_payload
        )

        if result["success"]:
            logger.info(f"Collaboration request sent to {target_agent_id}")

        return result

    except Exception as e:
        logger.error(f"Error requesting collaboration from {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def share_task_result(target_agent_id: str, task_id: str, result_data: Dict[str, Any],
                     meu_state_update: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Share task results with another agent

    Args:
        target_agent_id: ID of the agent to share results with
        task_id: Identifier of the completed task
        result_data: Results and artifacts from task completion
        meu_state_update: MEU workspace state updates

    Returns:
        Dict containing acknowledgment from target agent
    """
    try:
        result_payload = {
            "task_id": task_id,
            "completion_status": "completed",
            "results": result_data,
            "meu_state_update": meu_state_update or {},
            "timestamp": datetime.utcnow().isoformat(),
            "artifacts": result_data.get("artifacts", [])
        }

        return _a2a_client.send_message(
            target_agent_id,
            "task_result",
            result_payload
        )

    except Exception as e:
        logger.error(f"Error sharing task result with {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def query_agent_capabilities(target_agent_id: str) -> Dict[str, Any]:
    """
    Query specific capabilities of another agent

    Args:
        target_agent_id: ID of the agent to query

    Returns:
        Dict containing detailed capability information
    """
    try:
        query_payload = {
            "query_type": "capabilities",
            "requested_details": [
                "supported_protocols",
                "available_tools",
                "processing_capacity",
                "specialization_areas"
            ]
        }

        result = _a2a_client.send_message(
            target_agent_id,
            "capability_query",
            query_payload
        )

        return result

    except Exception as e:
        logger.error(f"Error querying capabilities of {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def share_git_progress(target_agent_id: str, progress_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Share Git/development progress with another agent

    Args:
        target_agent_id: ID of the agent to share progress with
        progress_type: Type of progress (commit, test_results, deployment, etc.)
        details: Progress details including commit info, test results, etc.

    Returns:
        Dict containing acknowledgment from target agent
    """
    try:
        progress_payload = {
            "progress_type": progress_type,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_role": os.getenv("PROJECT_ROLE", "developer"),
            "details": details,
            "coordination_needed": details.get("coordination_needed", False)
        }

        result = _a2a_client.send_message(
            target_agent_id,
            "git_progress",
            progress_payload
        )

        if result["success"]:
            logger.info(f"Git progress shared with {target_agent_id}: {progress_type}")

        return result

    except Exception as e:
        logger.error(f"Error sharing git progress with {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def coordinate_development_task(target_agent_id: str, task_type: str,
                               task_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate a development task with another agent

    Args:
        target_agent_id: ID of the agent to coordinate with
        task_type: Type of development task (backend, frontend, testing, etc.)
        task_details: Details about the task requirements and coordination

    Returns:
        Dict containing coordination response
    """
    try:
        coordination_payload = {
            "task_type": task_type,
            "task_details": task_details,
            "coordination_type": "development_sync",
            "sender_role": os.getenv("PROJECT_ROLE", "developer"),
            "timestamp": datetime.utcnow().isoformat(),
            "requires_response": task_details.get("requires_response", True)
        }

        result = _a2a_client.send_message(
            target_agent_id,
            "development_coordination",
            coordination_payload
        )

        if result["success"]:
            logger.info(f"Development coordination sent to {target_agent_id}: {task_type}")

        return result

    except Exception as e:
        logger.error(f"Error coordinating development task with {target_agent_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_agent": target_agent_id
        }


def broadcast_deployment_status(deployment_status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast deployment status to all discovered agents

    Args:
        deployment_status: Status of deployment (success, failed, in_progress)
        details: Deployment details including URLs, commit hash, etc.

    Returns:
        Dict containing broadcast results
    """
    try:
        # First discover all agents
        agents_result = discover_agents()
        if not agents_result["success"]:
            return {
                "success": False,
                "error": "Failed to discover agents for broadcast"
            }

        current_agent_id = os.getenv("AGENT_ID", "meu-coder")
        target_agents = [
            agent["agent_id"] for agent in agents_result["agents"]
            if agent["agent_id"] != current_agent_id
        ]

        if not target_agents:
            return {
                "success": True,
                "message": "No other agents to broadcast to",
                "broadcast_count": 0
            }

        broadcast_payload = {
            "deployment_status": deployment_status,
            "deployment_details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "sender_role": os.getenv("PROJECT_ROLE", "developer")
        }

        results = []
        for agent_id in target_agents:
            result = _a2a_client.send_message(
                agent_id,
                "deployment_broadcast",
                broadcast_payload
            )
            results.append({
                "agent_id": agent_id,
                "success": result["success"],
                "error": result.get("error")
            })

        successful_broadcasts = len([r for r in results if r["success"]])

        return {
            "success": True,
            "broadcast_count": successful_broadcasts,
            "total_agents": len(target_agents),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error broadcasting deployment status: {e}")
        return {
            "success": False,
            "error": str(e)
        }