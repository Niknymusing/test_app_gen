"""
MCP (Model Context Protocol) Integration Tools

This module implements MCP protocol integration for sharing context,
resources, and workspace state between MEU framework agents.
"""

import json
import logging
import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for MCP protocol communication with Streamable HTTP support"""

    def __init__(self, filesystem_uri: str = None, docs_uri: str = None):
        self.filesystem_uri = filesystem_uri or os.getenv("MCP_FILESYSTEM_URI", "http://mcp-filesystem:8200")
        self.docs_uri = docs_uri or os.getenv("MCP_DOCS_URI", "http://mcp-docs:8201")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MEU-Coder-Agent/1.0",
            "Accept": "application/json, text/plain",
            "Cache-Control": "no-cache"
        })

    def _streamable_http_request(self, method: str, url: str, data: dict = None) -> Dict[str, Any]:
        """Make a streamable HTTP request compatible with MCP specification"""
        try:
            # Add MCP-specific headers for streamable HTTP transport
            headers = {
                "X-MCP-Version": "2025-03-26",
                "X-MCP-Transport": "streamable_http"
            }

            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Handle streaming response if applicable
            if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                # Handle Server-Sent Events for streaming
                content = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        content += line[6:] + "\n"
                try:
                    return json.loads(content) if content.strip() else {}
                except json.JSONDecodeError:
                    return {"content": content}
            else:
                return response.json() if response.content else {}

        except Exception as e:
            logger.error(f"Streamable HTTP request failed: {e}")
            return {"success": False, "error": str(e)}

    def get_resources(self, resource_type: str = "all") -> Dict[str, Any]:
        """Get available resources from MCP filesystem server using Streamable HTTP"""
        try:
            # Use streamable HTTP transport
            data = self._streamable_http_request("GET", f"{self.filesystem_uri}/resources")

            if data.get("success", True) and "error" not in data:  # Assume success if not explicitly failed
                resources = data.get("resources", [])

                # Filter by type if specified
                if resource_type != "all":
                    resources = [r for r in resources if r.get("type") == resource_type]

                return {
                    "success": True,
                    "resources": resources,
                    "total_count": len(resources),
                    "server_uri": self.filesystem_uri,
                    "transport": "streamable_http"
                }
            else:
                return {
                    "success": False,
                    "error": data.get("error", "MCP server returned error"),
                    "server_uri": self.filesystem_uri
                }
        except Exception as e:
            logger.error(f"Failed to get MCP resources: {e}")
            return {
                "success": False,
                "error": str(e),
                "server_uri": self.filesystem_uri
            }

    def get_resource_content(self, resource_id: str) -> Dict[str, Any]:
        """Get content of a specific resource"""
        try:
            response = self.session.get(
                f"{self.filesystem_uri}/resources/{resource_id}",
                timeout=10
            )
            if response.status_code == 200:
                return {
                    "success": True,
                    "resource": response.json(),
                    "resource_id": resource_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Resource not found or server error: {response.status_code}",
                    "resource_id": resource_id
                }
        except Exception as e:
            logger.error(f"Failed to get resource content: {e}")
            return {
                "success": False,
                "error": str(e),
                "resource_id": resource_id
            }

    def inject_context(self, resource_ids: List[str]) -> Dict[str, Any]:
        """Inject context from multiple resources"""
        try:
            payload = {"resource_ids": resource_ids}
            response = self.session.post(
                f"{self.filesystem_uri}/context/inject",
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "context": data.get("context", {}),
                    "resource_count": data.get("resource_count", 0),
                    "timestamp": data.get("timestamp"),
                    "injected_resources": resource_ids
                }
            else:
                return {
                    "success": False,
                    "error": f"Context injection failed: {response.status_code}",
                    "resource_ids": resource_ids
                }
        except Exception as e:
            logger.error(f"Failed to inject context: {e}")
            return {
                "success": False,
                "error": str(e),
                "resource_ids": resource_ids
            }

    def get_documentation(self) -> Dict[str, Any]:
        """Get available documentation from MCP docs server"""
        try:
            response = self.session.get(f"{self.docs_uri}/docs", timeout=10)
            if response.status_code == 200:
                return {
                    "success": True,
                    "documentation": response.json(),
                    "docs_uri": self.docs_uri
                }
            else:
                return {
                    "success": False,
                    "error": f"Docs server returned status {response.status_code}",
                    "docs_uri": self.docs_uri
                }
        except Exception as e:
            logger.error(f"Failed to get documentation: {e}")
            return {
                "success": False,
                "error": str(e),
                "docs_uri": self.docs_uri
            }


# Global MCP client instance
_mcp_client = MCPClient()


def get_resources(resource_type: str = "all", file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get available resources from MCP filesystem server

    Args:
        resource_type: Type of resources to retrieve ("all", "file", "directory")
        file_extensions: Filter by file extensions (e.g., [".py", ".md"])

    Returns:
        Dict containing list of available resources
    """
    try:
        result = _mcp_client.get_resources(resource_type)

        if result["success"] and file_extensions:
            # Filter by file extensions
            filtered_resources = []
            for resource in result["resources"]:
                resource_path = resource.get("path", "")
                if any(resource_path.endswith(ext) for ext in file_extensions):
                    filtered_resources.append(resource)

            result["resources"] = filtered_resources
            result["total_count"] = len(filtered_resources)
            result["filtered_by_extensions"] = file_extensions

        return result

    except Exception as e:
        logger.error(f"Error in get_resources: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_resource_content(resource_id: str, parse_content: bool = True) -> Dict[str, Any]:
    """
    Get content of a specific resource with optional parsing

    Args:
        resource_id: ID/path of the resource to retrieve
        parse_content: Whether to parse content for analysis

    Returns:
        Dict containing resource content and metadata
    """
    try:
        result = _mcp_client.get_resource_content(resource_id)

        if result["success"] and parse_content:
            # Enhance with content analysis
            resource = result["resource"]
            content = resource.get("content", "")

            if content:
                analysis = {
                    "line_count": len(content.splitlines()),
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "is_code": _is_code_file(resource_id),
                    "language": _detect_language(resource_id),
                    "size_category": _categorize_size(len(content))
                }

                if analysis["is_code"]:
                    analysis.update(_analyze_code_content(content))

                result["content_analysis"] = analysis

        return result

    except Exception as e:
        logger.error(f"Error getting resource content: {e}")
        return {
            "success": False,
            "error": str(e),
            "resource_id": resource_id
        }


def inject_context(resource_ids: List[str], context_type: str = "meu_workspace") -> Dict[str, Any]:
    """
    Inject context from multiple resources for MEU workspace

    Args:
        resource_ids: List of resource IDs to inject
        context_type: Type of context injection

    Returns:
        Dict containing injected context data
    """
    try:
        result = _mcp_client.inject_context(resource_ids)

        if result["success"]:
            # Enhance context for MEU framework
            context = result["context"]
            enhanced_context = {
                "injected_at": datetime.utcnow().isoformat(),
                "context_type": context_type,
                "resources": {},
                "meu_metadata": {
                    "total_resources": len(resource_ids),
                    "resource_types": {},
                    "code_files": [],
                    "documentation_files": [],
                    "configuration_files": []
                }
            }

            for resource_id, content in context.items():
                enhanced_context["resources"][resource_id] = {
                    "content": content,
                    "type": _detect_file_type(resource_id),
                    "language": _detect_language(resource_id),
                    "size": len(content)
                }

                # Categorize files for MEU framework
                file_type = enhanced_context["resources"][resource_id]["type"]
                if file_type == "code":
                    enhanced_context["meu_metadata"]["code_files"].append(resource_id)
                elif file_type == "documentation":
                    enhanced_context["meu_metadata"]["documentation_files"].append(resource_id)
                elif file_type == "configuration":
                    enhanced_context["meu_metadata"]["configuration_files"].append(resource_id)

                # Update type counts
                enhanced_context["meu_metadata"]["resource_types"][file_type] = \
                    enhanced_context["meu_metadata"]["resource_types"].get(file_type, 0) + 1

            result["enhanced_context"] = enhanced_context

        return result

    except Exception as e:
        logger.error(f"Error injecting context: {e}")
        return {
            "success": False,
            "error": str(e),
            "resource_ids": resource_ids
        }


def share_meu_context(triplet_id: str, workspace_state: Dict[str, Any],
                     target_resources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Share MEU framework context and workspace state

    Args:
        triplet_id: MEU triplet identifier
        workspace_state: Current MEU workspace state
        target_resources: Specific resources to share

    Returns:
        Dict containing shared context information
    """
    try:
        # Prepare MEU-specific context
        meu_context = {
            "triplet_id": triplet_id,
            "workspace_state": workspace_state,
            "timestamp": datetime.utcnow().isoformat(),
            "framework_version": "MEU-1.0",
            "context_type": "meu_framework"
        }

        if target_resources:
            # Inject specific resources
            resource_result = inject_context(target_resources, "meu_share")
            if resource_result["success"]:
                meu_context["shared_resources"] = resource_result["enhanced_context"]

        # In a real implementation, this would push to an MCP server
        # For now, we'll structure it for sharing
        return {
            "success": True,
            "shared_context": meu_context,
            "triplet_id": triplet_id,
            "sharing_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error sharing MEU context: {e}")
        return {
            "success": False,
            "error": str(e),
            "triplet_id": triplet_id
        }


def sync_workspace_state(triplet_id: str, state_updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize MEU workspace state across agents

    Args:
        triplet_id: MEU triplet identifier
        state_updates: State changes to synchronize

    Returns:
        Dict containing synchronization result
    """
    try:
        sync_payload = {
            "triplet_id": triplet_id,
            "state_updates": state_updates,
            "sync_timestamp": datetime.utcnow().isoformat(),
            "sync_type": "meu_workspace"
        }

        # In a real implementation, this would sync with MCP servers
        # For now, we'll prepare the sync data structure
        return {
            "success": True,
            "sync_data": sync_payload,
            "sync_status": "prepared",
            "triplet_id": triplet_id
        }

    except Exception as e:
        logger.error(f"Error syncing workspace state: {e}")
        return {
            "success": False,
            "error": str(e),
            "triplet_id": triplet_id
        }


def _is_code_file(resource_id: str) -> bool:
    """Check if resource is a code file"""
    code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php']
    return any(resource_id.endswith(ext) for ext in code_extensions)


def _detect_language(resource_id: str) -> str:
    """Detect programming language from file extension"""
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c_header',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.md': 'markdown',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css'
    }

    for ext, lang in extension_map.items():
        if resource_id.endswith(ext):
            return lang

    return 'unknown'


def _detect_file_type(resource_id: str) -> str:
    """Detect general file type"""
    if _is_code_file(resource_id):
        return "code"
    elif resource_id.endswith(('.md', '.txt', '.rst')):
        return "documentation"
    elif resource_id.endswith(('.yml', '.yaml', '.json', '.toml', '.ini')):
        return "configuration"
    elif resource_id.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
        return "image"
    else:
        return "other"


def _categorize_size(size: int) -> str:
    """Categorize file size"""
    if size < 1000:
        return "small"
    elif size < 10000:
        return "medium"
    elif size < 100000:
        return "large"
    else:
        return "very_large"


def _analyze_code_content(content: str) -> Dict[str, Any]:
    """Analyze code content for additional insights"""
    lines = content.splitlines()
    return {
        "total_lines": len(lines),
        "blank_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith('#')),
        "has_imports": any('import ' in line for line in lines[:20]),
        "has_classes": 'class ' in content,
        "has_functions": 'def ' in content or 'function ' in content,
        "complexity_estimate": min(content.count('if ') + content.count('for ') + content.count('while '), 50)
    }