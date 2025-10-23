"""
MEU Framework Coder Agent - Correct Implementation

Following the MEU specification:
- M: Context available to Claude Code (persona, intention, task, tools, source code)
- E: Claude Code execution in Docker filesystem with logging
- U: A2A protocol input for evaluation/feedback

No separate LLM calls for domains - single Claude Code execution with full MEU context.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import subprocess
from pathlib import Path
import re
import json
import yaml
from resources.prompt_builder.base import PromptBuilder

# Optional imports - will gracefully handle missing tools
try:
    from resources.tools.a2a.a2a_tools import receive_evaluation, send_collaboration_request
except ImportError:
    def receive_evaluation(agent_id): return {"success": False, "error": "A2A tools not available"}
    def send_collaboration_request(data): return {"success": False, "error": "A2A tools not available"}

try:
    from resources.tools.mcp.mcp_tools import get_resources, sync_workspace_state
except ImportError:
    def get_resources(): return {"success": False, "resources": [], "error": "MCP tools not available"}
    def sync_workspace_state(triplet_id, data): return {"success": False, "error": "MCP tools not available"}

try:
    from resources.tools.meu.meu_tools import create_triplet, update_workspace_state
except ImportError:
    def create_triplet(source, data): return {"success": False, "triplet_id": None, "error": "MEU tools not available"}
    def update_workspace_state(data, triplet_id): return {"success": False, "error": "MEU tools not available"}

try:
    from resources.tools.claude_code.claude_code_tools import ClaudeCodeIntegration
except ImportError:
    ClaudeCodeIntegration = None

import anthropic
from anthropic import Anthropic

logger = logging.getLogger(__name__)


def run_claude_command(command_args: list = None, prompt: str = "", working_directory: str = "/workspace") -> Dict[str, Any]:
    """Execute Claude Code CLI directly within the container environment"""
    try:
        # Check if Claude Code is available
        check_result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
        if check_result.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Claude Code CLI not available",
                "returncode": -1
            }

        # Build command with proper flags for container environment
        cmd = ["claude"] + (command_args or [])
        if prompt:
            cmd.append(prompt)

            # Execute Claude Code
            result = subprocess.run(
                cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(cmd)
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Claude Code execution timed out after 5 minutes",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


class MEUCoderAgentGraph:
    """
    MEU Coder Agent following the correct MEU specification.

    M: Builds context for Claude Code
    E: Executes Claude Code once with full context
    U: Receives A2A evaluation feedback and updates M domain via R,R* arrows
    """

    def __init__(self, workspace_path: str = "/workspace", agent_id: str = "meu-coder"):
        self.workspace_path = workspace_path
        self.agent_id = agent_id
        self.logger = logging.getLogger(__name__)
        self.model_context_base = {
            "performance_feedback": [],
            "optimization_patterns": [],
            "successful_strategies": [],
            "error_patterns_to_avoid": [],
            "context_adaptations": []
        }
        self.prompt_builder = PromptBuilder()
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        prompt_path = Path("/app/data/prompts/meu_execute/meu_execute.yaml")  # Use /app (Docker image) not /workspace (mounted volume)
        with open(prompt_path, "r") as f:
            self.prompts["meu_execute"] = yaml.safe_load(f)

    async def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the MEU workflow: Model -> Execute -> Update
        """
        self.user_request = initial_state["user_request"]
        context = {
            "workspace_path": initial_state.get("workspace_path", self.workspace_path),
            "task_context": initial_state.get("context", {}),
            "priority": initial_state.get("priority", "normal"),
            "task_id": initial_state.get("task_id")
        }

        return await self.execute_meu_workflow(self.user_request, context)

    async def execute_meu_workflow(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the MEU workflow: Model -> Execute -> Update
        """
        try:
            self.logger.info("Starting MEU workflow execution")

            # Create MEU triplet for tracking
            triplet_result = create_triplet("source", {
                "user_request": user_request,
                "agent_id": self.agent_id,
                "workspace_path": self.workspace_path
            })

            triplet_id = triplet_result.get("triplet_id") if triplet_result["success"] else None

            # M: Model Domain - Build context for Claude Code
            model_context = await self._build_model_context(user_request, context or {})

            # E: Execute Domain - Single Claude Code execution
            execution_result = await self._execute_claude_code(model_context)

            # U: Update Domain - Process A2A evaluation feedback
            update_result = await self._process_update_feedback(execution_result, triplet_id)

            # Final result
            final_result = {
                "triplet_id": triplet_id,
                "model_context": model_context,
                "execution_result": execution_result,
                "update_result": update_result,
                "status": "completed" if execution_result.get("success") else "failed",
                "timestamp": datetime.utcnow().isoformat()
            }

            # Update workspace state
            if triplet_id:
                update_workspace_state({
                    "operation": "meu_workflow_complete",
                    "data": final_result
                }, triplet_id)

            self.logger.info("MEU workflow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"MEU workflow failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _build_model_context(self, user_request: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        M: Model Domain - Build complete context for Claude Code execution

        Includes: agent persona, intention, task description, available tools, source code
        """
        self.logger.info("Building model context (M domain)")

        # Agent persona and intention
        agent_persona = {
            "role": "MEU Framework Coder Agent",
            "capabilities": [
                "Full-stack software development",
                "Claude Code CLI integration",
                "MCP tool integration",
                "A2A agent collaboration",
                "Test-driven development",
                "CI/CD pipeline setup"
            ],
            "working_environment": "Docker container with filesystem access",
            "tools_available": ["Claude Code CLI", "Git", "Node.js", "Python"]
        }

        # Get available MCP tools/resources
        mcp_resources = get_resources()
        available_tools = mcp_resources.get("resources", []) if mcp_resources.get("success") else []

        # Get current workspace state
        workspace_files = []
        try:
            import os
            for root, dirs, files in os.walk(self.workspace_path):
                for file in files:
                    if not file.startswith('.') and not 'logs' in root:
                        rel_path = os.path.relpath(os.path.join(root, file), self.workspace_path)
                        workspace_files.append(rel_path)
        except Exception as e:
            self.logger.warning(f"Could not scan workspace: {e}")

        # Build complete context with persistent M domain feedback (R,R* arrows integration)
        model_context = {
            "agent_persona": agent_persona,
            "task_intention": f"Implement user request: {user_request}",
            "user_request": user_request,
            "additional_context": additional_context,
            "available_tools": {
                "mcp_resources": available_tools,
                "workspace_path": self.workspace_path,
                "current_files": workspace_files
            },
            "execution_environment": {
                "type": "docker_container",
                "working_directory": self.workspace_path,
                "claude_code_available": True,
                "git_available": True
            },
            # R,R* arrows: M domain updated with U domain feedback
            "performance_optimization": {
                "previous_feedback": self.model_context_base["performance_feedback"],
                "successful_patterns": self.model_context_base["successful_strategies"],
                "patterns_to_avoid": self.model_context_base["error_patterns_to_avoid"],
                "context_adaptations": self.model_context_base["context_adaptations"]
            },
            "context_timestamp": datetime.utcnow().isoformat()
        }

        self.logger.info(f"Model context built with {len(available_tools)} MCP resources")
        return model_context

    async def _execute_claude_code(self, model_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        E: Execute Domain - Use Anthropic API to generate code and execute it
        """
        self.logger.info("Executing Claude API call for code generation (E domain)")

        claude_prompt = self._build_claude_prompt(model_context)
        self.logger.info(f"Claude Prompt:\\n{claude_prompt}")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "success": False,
                "stdout": "",
                "stderr": "ANTHROPIC_API_KEY not set",
                "returncode": -1
            }

        try:
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": claude_prompt}
                ]
            )
            response_content = message.content[0].text if message.content else ""

            # Extract code from response (assume it's a Python script)
            code_match = re.search(r'```python\n(.*?)\n```', response_content, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = response_content  # Fallback

            # Save generated code to file
            script_path = os.path.join(self.workspace_path, "generated_script.py")
            with open(script_path, "w") as f:
                f.write(code)

            # Execute the generated script
            execution_result = subprocess.run(
                ["python", script_path],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=60,
                env=os.environ.copy()
            )

            self.logger.info(f"Generated code:\\n{code}")
            self.logger.info(f"Execution result: success={execution_result.returncode == 0}, stdout={execution_result.stdout}, stderr={execution_result.stderr}")
            return {
                "success": execution_result.returncode == 0,
                "stdout": execution_result.stdout,
                "stderr": execution_result.stderr,
                "returncode": execution_result.returncode,
                "generated_code": code,
                "script_path": script_path,
                "response_content": response_content
            }
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def _build_claude_prompt(self, model_context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for Claude Code with full MEU context"""
        prompt_yaml = self.prompts["meu_execute"]
        format_args = {
            "user_request": self.user_request,
            "model_context": json.dumps(model_context, indent=2),
            "workspace_path": self.workspace_path
        }
        messages = self.prompt_builder.build_from_yaml(
            prompt_yaml,
            format_args=format_args,
            as_open_ai_messages=True
        )

        # Add explicit instruction for code generation
        messages.append({
            "role": "system",
            "content": "Generate a complete, executable Python script that precisely fulfills the user's request. The script should be self-contained and handle all necessary operations."
        })

        prompt = json.dumps(messages)
        return prompt

    async def _process_update_feedback(self, execution_result: Dict[str, Any], triplet_id: Optional[str]) -> Dict[str, Any]:
        """
        U: Update Domain - Process evaluation feedback via A2A protocol and update M domain (R,R* arrows)
        """
        self.logger.info("Processing update feedback (U domain)")

        # Evaluate execution results
        execution_success = execution_result.get("success", False)
        stdout_content = execution_result.get("stdout", "")
        stderr_content = execution_result.get("stderr", "")

        update_result = {
            "self_evaluation": {
                "claude_code_executed": execution_success,
                "stdout_length": len(stdout_content),
                "stderr_present": bool(stderr_content),
                "execution_time": "completed",
                "files_likely_created": "yes" if execution_success else "unknown"
            },
            "a2a_feedback": None,  # Would be populated by external A2A evaluation
            "update_timestamp": datetime.utcnow().isoformat()
        }

        # Try to get A2A feedback (will likely fail in current setup, but shows the pattern)
        try:
            a2a_feedback = receive_evaluation(self.agent_id)
            if a2a_feedback.get("success"):
                update_result["a2a_feedback"] = a2a_feedback
                self.logger.info("Received A2A evaluation feedback")
        except Exception as e:
            self.logger.debug(f"No A2A feedback available: {e}")

        # R,R* arrows: Update M domain based on U domain feedback
        await self._update_model_context_from_feedback(update_result, execution_result)

        # Sync with MCP if available
        if triplet_id:
            sync_result = sync_workspace_state(triplet_id, {
                "execution_result": execution_result,
                "update_result": update_result
            })
            update_result["mcp_sync"] = sync_result.get("success", False)

        return update_result

    async def _update_model_context_from_feedback(self, update_result: Dict[str, Any], execution_result: Dict[str, Any]):
        """
        R,R* arrows: Update M domain context based on U domain evaluation feedback
        This implements the MEU framework's feedback loop for continuous improvement
        """
        self.logger.info("Updating M domain context via R,R* arrows")

        execution_success = execution_result.get("success", False)
        stdout_content = execution_result.get("stdout", "")
        stderr_content = execution_result.get("stderr", "")

        # Record feedback for future context optimization
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_success": execution_success,
            "stdout_length": len(stdout_content),
            "stderr_present": bool(stderr_content),
            "a2a_feedback_available": update_result.get("a2a_feedback") is not None
        }

        self.model_context_base["performance_feedback"].append(feedback_entry)

        # Update successful strategies
        if execution_success:
            strategy = {
                "timestamp": datetime.utcnow().isoformat(),
                "context_patterns": "successful_claude_code_execution",
                "execution_indicators": {
                    "clean_execution": not bool(stderr_content),
                    "substantial_output": len(stdout_content) > 100
                }
            }
            self.model_context_base["successful_strategies"].append(strategy)

        # Update error patterns to avoid
        if not execution_success or stderr_content:
            error_pattern = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_indicators": {
                    "execution_failed": not execution_success,
                    "stderr_present": bool(stderr_content),
                    "stderr_snippet": stderr_content[:200] if stderr_content else None
                }
            }
            self.model_context_base["error_patterns_to_avoid"].append(error_pattern)

        # Add context adaptations based on feedback
        if update_result.get("a2a_feedback"):
            adaptation = {
                "timestamp": datetime.utcnow().isoformat(),
                "adaptation_type": "a2a_feedback_integration",
                "feedback_data": update_result["a2a_feedback"]
            }
            self.model_context_base["context_adaptations"].append(adaptation)

        # Keep feedback history manageable (last 10 entries)
        for key in ["performance_feedback", "successful_strategies", "error_patterns_to_avoid", "context_adaptations"]:
            if len(self.model_context_base[key]) > 10:
                self.model_context_base[key] = self.model_context_base[key][-10:]

        self.logger.info(f"M domain updated with {len(self.model_context_base['performance_feedback'])} feedback entries")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status for API compatibility"""
        return {
            "agent_id": self.agent_id,
            "capabilities": ["MEU Framework", "Claude Code Integration", "A2A Protocol", "MCP Integration"],
            "domains": ["Model Context Building", "Claude Code Execution", "A2A Evaluation"],
            "tools": ["Claude Code CLI", "Git", "Docker", "MCP Tools"],
            "status": "ready"
        }

    def add_message(self, role: str, content: str, message_id: str = None) -> Dict[str, Any]:
        """Add a message to the A2A-compliant message state"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": message_id or f"msg_{len(self.messages)}"
        }
        self.messages.append(message)
        return message

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in A2A-compliant format"""
        return self.messages

    def process_a2a_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming A2A protocol message"""
        try:
            # Extract message content from A2A format
            message = message_data.get("message", {})
            role = message.get("role", "user")

            # Handle text parts from A2A protocol
            parts = message.get("parts", [])
            content_parts = []
            for part in parts:
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))

            content = " ".join(content_parts) if content_parts else message.get("content", "")

            # Add to message state
            self.add_message(role, content, message_data.get("messageId"))

            return {
                "success": True,
                "processed_content": content,
                "message_count": len(self.messages)
            }
        except Exception as e:
            self.logger.error(f"Failed to process A2A message: {e}")
            return {
                "success": False,
                "error": str(e)
            }