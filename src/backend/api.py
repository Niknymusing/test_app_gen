"""
MEU Framework Coder Agent - FastAPI Backend

This module implements the FastAPI backend for the MEU coder agent,
providing REST API endpoints for agent interaction and monitoring.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from meu_agent import MEUCoderAgentGraph

logger = logging.getLogger(__name__)


# Pydantic models for API
class TaskRequest(BaseModel):
    """Request model for task execution."""
    user_request: str = Field(..., description="User's request or requirement")
    workspace_path: Optional[str] = Field("/workspace", description="Workspace path")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    priority: Optional[str] = Field("normal", description="Task priority: low, normal, high")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    created_at: str = Field(..., description="Task creation timestamp")


class TaskStatus(BaseModel):
    """Model for task status information."""
    task_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    current_stage: str
    started_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class A2AMessage(BaseModel):
    """Model for A2A protocol messages."""
    sender_id: str
    target_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None


class A2ATextPart(BaseModel):
    """Model for A2A text parts."""
    kind: str = "text"
    text: str


class A2AMessageContent(BaseModel):
    """Model for A2A message content."""
    role: str
    parts: List[A2ATextPart]


class A2ARequest(BaseModel):
    """Model for A2A JSON-RPC requests."""
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Dict[str, Any]


class AgentCapabilities(BaseModel):
    """Model for agent capabilities."""
    agent_id: str
    capabilities: List[str]
    domains: List[str]
    tools: List[str]
    status: str


# MCP Schema Definitions
class MCPInputState(TypedDict):
    """MCP-compliant input schema for software development tasks."""
    task_description: str
    workspace_path: Optional[str]
    priority: Optional[str]


class MCPOutputState(TypedDict):
    """MCP-compliant output schema for software development results."""
    task_id: str
    implementation_status: str
    files_created: List[str]
    execution_summary: str


class MCPToolRequest(BaseModel):
    """Model for MCP tool execution requests."""
    tool_name: str
    input_data: MCPInputState
    session_id: Optional[str] = None


class MCPToolResponse(BaseModel):
    """Model for MCP tool execution responses."""
    tool_name: str
    output_data: MCPOutputState
    execution_time: float
    success: bool
    session_id: Optional[str] = None




# Global state for tracking tasks
active_tasks: Dict[str, Dict[str, Any]] = {}


def create_app(agent_graph: MEUCoderAgentGraph) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        agent_graph: Initialized MEU coder agent graph

    Returns:
        Configured FastAPI application
    """

    app = FastAPI(
        title="MEU Coder Agent API",
        description="RESTful API for MEU Framework Coder Agent with Claude Code integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependency to get agent graph
    def get_agent_graph() -> MEUCoderAgentGraph:
        return agent_graph

    @app.get("/")
    async def root():
        """Root endpoint with agent information."""
        return {
            "agent_type": "meu-coder",
            "version": "1.0.0",
            "status": "running",
            "framework": "MEU (Model-Execute-Update)",
            "tools": ["Claude Code", "A2A", "MCP"],
            "timestamp": datetime.utcnow().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_tasks": len(active_tasks)
        }

    @app.get("/capabilities", response_model=AgentCapabilities)
    async def get_capabilities(graph: MEUCoderAgentGraph = Depends(get_agent_graph)):
        """Get agent capabilities and status."""
        status_info = graph.get_workflow_status()
        return AgentCapabilities(**status_info)

    @app.post("/api/v1/tasks", response_model=TaskResponse)
    async def create_task(
        request: TaskRequest,
        background_tasks: BackgroundTasks,
        graph: MEUCoderAgentGraph = Depends(get_agent_graph)
    ):
        """Create and execute a new task."""
        try:
            task_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            # Create task state
            task_state = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0.0,
                "current_stage": "initialization",
                "started_at": timestamp,
                "updated_at": timestamp,
                "request": request.dict(),
                "result": None,
                "error": None
            }

            active_tasks[task_id] = task_state

            # Queue task for background execution
            background_tasks.add_task(
                execute_task_background,
                task_id,
                request,
                graph
            )

            logger.info(f"Created task {task_id}: {request.user_request[:100]}")

            return TaskResponse(
                task_id=task_id,
                status="queued",
                message="Task queued for execution",
                created_at=timestamp
            )

        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/tasks/{task_id}", response_model=TaskStatus)
    async def get_task_status(task_id: str):
        """Get status of a specific task."""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task_state = active_tasks[task_id]
        return TaskStatus(**task_state)

    @app.get("/api/v1/tasks", response_model=List[TaskStatus])
    async def list_tasks(status: Optional[str] = None, limit: int = 50):
        """List active tasks with optional status filter."""
        tasks = list(active_tasks.values())

        if status:
            tasks = [task for task in tasks if task["status"] == status]

        # Sort by creation time (most recent first)
        tasks.sort(key=lambda x: x["started_at"], reverse=True)

        return [TaskStatus(**task) for task in tasks[:limit]]

    @app.delete("/api/v1/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """Cancel a task (if possible)."""
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task_state = active_tasks[task_id]

        if task_state["status"] in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed task")

        task_state["status"] = "cancelled"
        task_state["updated_at"] = datetime.utcnow().isoformat()

        return {"message": f"Task {task_id} cancelled", "status": "cancelled"}

    @app.post("/a2a/{assistant_id}")
    async def a2a_endpoint(
        assistant_id: str,
        request: A2ARequest,
        graph: MEUCoderAgentGraph = Depends(get_agent_graph)
    ):
        """A2A protocol endpoint for agent-to-agent communication."""
        try:
            if request.method == "message/send":
                # Process A2A message
                params = request.params
                message_data = params.get("message", {})

                # Process the message through the agent
                result = graph.process_a2a_message({
                    "message": message_data,
                    "messageId": params.get("messageId", ""),
                    "thread": params.get("thread", {})
                })

                if result.get("success"):
                    # Create response in A2A format
                    response_content = result.get("processed_content", "Message processed")

                    return {
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "result": {
                            "artifacts": [{
                                "parts": [{
                                    "kind": "text",
                                    "text": f"Processed: {response_content}"
                                }]
                            }]
                        }
                    }
                else:
                    raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

        except Exception as e:
            logger.error(f"Failed to process A2A request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card(
        assistant_id: Optional[str] = None,
        graph: MEUCoderAgentGraph = Depends(get_agent_graph)
    ):
        """Return A2A agent card for discovery."""
        return {
            "name": "MEU Coder Agent",
            "description": "Advanced Model-Execute-Update framework agent with Claude Code integration for software development",
            "assistant_id": assistant_id or graph.agent_id,
            "capabilities": [
                "software-development",
                "code-generation",
                "testing",
                "debugging",
                "documentation"
            ],
            "input_modes": ["text"],
            "output_modes": ["text", "code"],
            "a2a_endpoint": f"/a2a/{assistant_id or graph.agent_id}",
            "version": "1.0.0",
            "protocols": ["A2A", "MCP"],
            "tools": ["Claude Code CLI", "Git", "Docker", "Python", "Node.js"]
        }

    @app.post("/mcp")
    async def mcp_endpoint(
        request: MCPToolRequest,
        graph: MEUCoderAgentGraph = Depends(get_agent_graph)
    ):
        """MCP protocol endpoint for tool exposure."""
        try:
            import time
            start_time = time.time()

            # Execute the MEU agent as an MCP tool
            initial_state = {
                "user_request": request.input_data["task_description"],
                "workspace_path": request.input_data.get("workspace_path", "/workspace"),
                "context": {
                    "priority": request.input_data.get("priority", "normal"),
                    "session_id": request.session_id,
                    "mcp_tool_execution": True
                }
            }

            # Execute through MEU framework
            result = await graph.execute(initial_state)
            execution_time = time.time() - start_time

            # Format response in MCP-compliant structure
            output_data = MCPOutputState(
                task_id=result.get("task_id", str(uuid.uuid4())),
                implementation_status=result.get("status", "completed"),
                files_created=result.get("files_created", []),
                execution_summary=result.get("execution_summary", "Task completed through MEU framework")
            )

            return MCPToolResponse(
                tool_name="meu-coder-agent",
                output_data=output_data,
                execution_time=execution_time,
                success=result.get("success", True),
                session_id=request.session_id
            )

        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/mcp/tools")
    async def list_mcp_tools():
        """List available MCP tools."""
        return {
            "tools": [{
                "name": "meu-coder-agent",
                "description": "Advanced software development agent using MEU framework",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the software development task"
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to the workspace directory",
                            "default": "/workspace"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "default": "normal"
                        }
                    },
                    "required": ["task_description"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "implementation_status": {"type": "string"},
                        "files_created": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "execution_summary": {"type": "string"}
                    }
                }
            }]
        }

    @app.post("/api/v1/a2a/message")
    async def receive_a2a_message(
        message: A2AMessage,
        graph: MEUCoderAgentGraph = Depends(get_agent_graph)
    ):
        """Receive A2A protocol message from another agent."""
        try:
            logger.info(f"Received A2A message from {message.sender_id}: {message.message_type}")

            # Handle different message types
            if message.message_type == "collaboration_request":
                response = await handle_collaboration_request(message, graph)
            elif message.message_type == "task_result":
                response = await handle_task_result(message)
            elif message.message_type == "capability_query":
                response = await handle_capability_query(graph)
            else:
                response = {
                    "success": False,
                    "error": f"Unknown message type: {message.message_type}"
                }

            return {
                "message_id": str(uuid.uuid4()),
                "response_to": message.sender_id,
                "timestamp": datetime.utcnow().isoformat(),
                "response": response
            }

        except Exception as e:
            logger.error(f"Failed to process A2A message: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/workspace/state")
    async def get_workspace_state():
        """Get current workspace state."""
        # This would typically read from MEU workspace state
        return {
            "workspace_path": agent_graph.workspace_path,
            "agent_id": agent_graph.agent_id,
            "active_tasks": len(active_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }

    @app.post("/api/v1/workspace/sync")
    async def sync_workspace_state(sync_data: Dict[str, Any]):
        """Sync workspace state via MCP."""
        try:
            from resources.tools.mcp import sync_workspace_state

            result = sync_workspace_state(
                sync_data.get("triplet_id", "unknown"),
                sync_data.get("state_updates", {})
            )

            return {
                "success": result["success"],
                "sync_status": result.get("sync_status", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Workspace sync failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def execute_task_background(task_id: str, request: TaskRequest, graph: MEUCoderAgentGraph):
    """Execute task in background."""
    try:
        logger.info(f"Starting background execution for task {task_id}")

        # Update task status
        task_state = active_tasks[task_id]
        task_state["status"] = "running"
        task_state["current_stage"] = "model_domain"
        task_state["progress"] = 0.1
        task_state["updated_at"] = datetime.utcnow().isoformat()

        # Prepare initial state for graph execution
        initial_state = {
            "user_request": request.user_request,
            "workspace_path": request.workspace_path,
            "context": request.context,
            "priority": request.priority,
            "task_id": task_id
        }

        # Execute Model domain
        task_state["current_stage"] = "model_domain"
        task_state["progress"] = 0.3

        # Execute the MEU workflow
        result = await graph.execute(initial_state)

        # Update task state based on execution result
        if result.get("error"):
            task_state["status"] = "failed"
            task_state["error"] = result["error"]
        else:
            task_state["status"] = "completed"
            task_state["result"] = {
                "final_status": result.get("final_status", "unknown"),
                "evaluation_summary": result.get("evaluation_summary"),
                "implementation_results": result.get("implementation_results"),
                "recommendations": result.get("recommendations")
            }

        task_state["progress"] = 1.0
        task_state["current_stage"] = "completed"
        task_state["updated_at"] = datetime.utcnow().isoformat()

        logger.info(f"Task {task_id} completed with status: {task_state['status']}")

    except Exception as e:
        logger.error(f"Task {task_id} execution failed: {e}")
        task_state = active_tasks.get(task_id, {})
        task_state.update({
            "status": "failed",
            "error": str(e),
            "progress": 0.0,
            "updated_at": datetime.utcnow().isoformat()
        })


async def handle_collaboration_request(message: A2AMessage, graph: MEUCoderAgentGraph) -> Dict[str, Any]:
    """Handle collaboration request from another agent."""
    try:
        payload = message.payload
        task_description = payload.get("task_description", "")

        # Create a collaborative task
        task_request = TaskRequest(
            user_request=f"Collaboration request: {task_description}",
            context={
                "collaboration": True,
                "requesting_agent": message.sender_id,
                "meu_context": payload.get("meu_context", {})
            }
        )

        # Execute the task
        task_id = str(uuid.uuid4())
        initial_state = {
            "user_request": task_request.user_request,
            "workspace_path": "/workspace",
            "context": task_request.context,
            "task_id": task_id,
            "collaboration_mode": True
        }

        result = await graph.execute(initial_state)

        return {
            "success": True,
            "collaboration_accepted": True,
            "task_id": task_id,
            "result": result,
            "agent_id": graph.agent_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "collaboration_accepted": False
        }


async def handle_task_result(message: A2AMessage) -> Dict[str, Any]:
    """Handle task result from another agent."""
    payload = message.payload
    task_id = payload.get("task_id", "unknown")

    logger.info(f"Received task result for task {task_id} from {message.sender_id}")

    return {
        "success": True,
        "acknowledged": True,
        "task_id": task_id
    }


async def handle_capability_query(graph: MEUCoderAgentGraph) -> Dict[str, Any]:
    """Handle capability query from another agent."""
    capabilities = graph.get_workflow_status()

    return {
        "success": True,
        "capabilities": capabilities,
        "agent_id": graph.agent_id,
        "protocols_supported": ["A2A", "MCP"],
        "tools_available": ["Claude Code", "MEU Framework"]
    }