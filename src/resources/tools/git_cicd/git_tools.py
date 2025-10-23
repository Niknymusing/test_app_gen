"""
Git and CI/CD Integration Tools for MEU Framework
Provides version control and deployment capabilities for collaborative development
"""

import subprocess
import os
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class GitCICDManager:
    """Manages Git operations and CI/CD workflows for MEU agents"""

    def __init__(self, workspace_path: str = "/workspace"):
        self.workspace_path = Path(workspace_path)
        self.repo_url = os.getenv("GIT_REPO_URL")
        self.branch = os.getenv("GIT_BRANCH", "main")
        self.token = os.getenv("GITHUB_TOKEN")
        self.user_name = os.getenv("GIT_USER_NAME", "MEU Agent")
        self.user_email = os.getenv("GIT_USER_EMAIL", "agent@meu.dev")
        self.agent_id = os.getenv("AGENT_ID", "meu-agent")
        self.project_role = os.getenv("PROJECT_ROLE", "developer")

    def _run_git_command(self, command: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Execute git command with proper authentication"""
        try:
            if cwd is None:
                cwd = self.workspace_path

            # Set up Git credentials for GitHub
            env = os.environ.copy()
            if self.token:
                # Use token authentication for HTTPS
                env["GIT_ASKPASS"] = "echo"
                env["GIT_USERNAME"] = "token"
                env["GIT_PASSWORD"] = self.token

            result = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                env=env,
                timeout=300
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "command": " ".join(command)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Git command timed out",
                "command": " ".join(command)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(command)
            }

    def initialize_repository(self) -> Dict[str, Any]:
        """Initialize or clone the repository"""
        try:
            repo_dir = self.workspace_path / "project"

            if repo_dir.exists() and (repo_dir / ".git").exists():
                logger.info("Repository already exists, pulling latest changes")
                return self.pull_changes()

            # Clone the repository
            logger.info(f"Cloning repository {self.repo_url}")

            # Use token in URL for authentication
            auth_url = self.repo_url.replace("https://", f"https://{self.token}@") if self.token else self.repo_url

            result = self._run_git_command([
                "git", "clone", auth_url, str(repo_dir)
            ])

            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to clone repository: {result.get('stderr', 'Unknown error')}"
                }

            # Configure Git user
            config_result = self.configure_git_user()
            if not config_result["success"]:
                logger.warning(f"Failed to configure git user: {config_result.get('error')}")

            # Configure remote with token authentication for pushing
            if self.token:
                remote_result = self._run_git_command([
                    "git", "remote", "set-url", "origin", auth_url
                ], repo_dir)
                if not remote_result["success"]:
                    logger.warning(f"Failed to set remote URL with token: {remote_result.get('error')}")

            return {
                "success": True,
                "message": "Repository cloned successfully",
                "repo_path": str(repo_dir)
            }

        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def configure_git_user(self) -> Dict[str, Any]:
        """Configure Git user for commits"""
        try:
            repo_dir = self.workspace_path / "project"

            # Set user name and email
            name_result = self._run_git_command([
                "git", "config", "user.name", self.user_name
            ], repo_dir)

            email_result = self._run_git_command([
                "git", "config", "user.email", self.user_email
            ], repo_dir)

            if name_result["success"] and email_result["success"]:
                return {
                    "success": True,
                    "message": f"Git user configured: {self.user_name} <{self.user_email}>"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to configure git user: {name_result.get('stderr')} {email_result.get('stderr')}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def create_branch(self, branch_name: str) -> Dict[str, Any]:
        """Create and switch to a new branch"""
        try:
            repo_dir = self.workspace_path / "project"

            # Create and checkout new branch
            result = self._run_git_command([
                "git", "checkout", "-b", branch_name
            ], repo_dir)

            if result["success"]:
                return {
                    "success": True,
                    "message": f"Created and switched to branch: {branch_name}",
                    "branch": branch_name
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create branch: {result.get('stderr')}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Commit changes to the repository"""
        try:
            repo_dir = self.workspace_path / "project"

            # Add files
            if files:
                for file in files:
                    add_result = self._run_git_command([
                        "git", "add", file
                    ], repo_dir)
                    if not add_result["success"]:
                        logger.warning(f"Failed to add file {file}: {add_result.get('stderr')}")
            else:
                # Add all changes
                add_result = self._run_git_command([
                    "git", "add", "."
                ], repo_dir)
                if not add_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to add files: {add_result.get('stderr')}"
                    }

            # Check if there are changes to commit
            status_result = self._run_git_command([
                "git", "status", "--porcelain"
            ], repo_dir)

            if not status_result["stdout"].strip():
                return {
                    "success": True,
                    "message": "No changes to commit",
                    "changes": False
                }

            # Commit changes
            commit_message = f"[{self.agent_id}] {message}\n\n🤖 Generated with MEU Framework\nAgent Role: {self.project_role}"

            commit_result = self._run_git_command([
                "git", "commit", "-m", commit_message
            ], repo_dir)

            if commit_result["success"]:
                # Get commit hash
                hash_result = self._run_git_command([
                    "git", "rev-parse", "HEAD"
                ], repo_dir)

                return {
                    "success": True,
                    "message": f"Changes committed successfully",
                    "commit_hash": hash_result.get("stdout", "")[:8],
                    "commit_message": commit_message,
                    "changes": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to commit: {commit_result.get('stderr')}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def push_changes(self, branch: Optional[str] = None) -> Dict[str, Any]:
        """Push changes to remote repository"""
        try:
            repo_dir = self.workspace_path / "project"
            target_branch = branch or self.branch

            # Push to remote
            push_result = self._run_git_command([
                "git", "push", "origin", target_branch
            ], repo_dir)

            if push_result["success"]:
                return {
                    "success": True,
                    "message": f"Changes pushed to {target_branch}",
                    "branch": target_branch
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to push: {push_result.get('stderr')}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def pull_changes(self) -> Dict[str, Any]:
        """Pull latest changes from remote"""
        try:
            repo_dir = self.workspace_path / "project"

            # Pull changes
            pull_result = self._run_git_command([
                "git", "pull", "origin", self.branch
            ], repo_dir)

            if pull_result["success"]:
                return {
                    "success": True,
                    "message": "Latest changes pulled successfully",
                    "output": pull_result.get("stdout", "")
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to pull: {pull_result.get('stderr')}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """Get repository status"""
        try:
            repo_dir = self.workspace_path / "project"

            # Get status
            status_result = self._run_git_command([
                "git", "status", "--porcelain"
            ], repo_dir)

            # Get current branch
            branch_result = self._run_git_command([
                "git", "branch", "--show-current"
            ], repo_dir)

            # Get last commit
            log_result = self._run_git_command([
                "git", "log", "--oneline", "-1"
            ], repo_dir)

            return {
                "success": True,
                "current_branch": branch_result.get("stdout", ""),
                "last_commit": log_result.get("stdout", ""),
                "changes": status_result.get("stdout", "").split('\n') if status_result.get("stdout") else [],
                "has_changes": bool(status_result.get("stdout", "").strip())
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_tests(self) -> Dict[str, Any]:
        """Run project tests"""
        try:
            repo_dir = self.workspace_path / "project"

            # Check for different test frameworks
            if (repo_dir / "package.json").exists():
                # Node.js project
                test_result = self._run_git_command([
                    "npm", "test"
                ], repo_dir)
            elif (repo_dir / "requirements.txt").exists() or (repo_dir / "pyproject.toml").exists():
                # Python project
                test_result = self._run_git_command([
                    "python", "-m", "pytest"
                ], repo_dir)
            else:
                return {
                    "success": False,
                    "error": "No recognized test framework found"
                }

            return {
                "success": test_result["success"],
                "output": test_result.get("stdout", ""),
                "error": test_result.get("stderr", "") if not test_result["success"] else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_git_manager = GitCICDManager()


def initialize_repository() -> Dict[str, Any]:
    """Initialize the Git repository"""
    return _git_manager.initialize_repository()


def commit_and_push(message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Commit and push changes"""
    commit_result = _git_manager.commit_changes(message, files)
    if not commit_result["success"] or not commit_result.get("changes", False):
        return commit_result

    push_result = _git_manager.push_changes()
    return {
        "success": push_result["success"],
        "commit": commit_result,
        "push": push_result
    }


def sync_with_remote() -> Dict[str, Any]:
    """Sync with remote repository"""
    return _git_manager.pull_changes()


def get_repository_status() -> Dict[str, Any]:
    """Get current repository status"""
    return _git_manager.get_status()


def run_project_tests() -> Dict[str, Any]:
    """Run project tests"""
    return _git_manager.run_tests()


def create_feature_branch(feature_name: str) -> Dict[str, Any]:
    """Create a feature branch for development"""
    branch_name = f"{_git_manager.agent_id}-{feature_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    return _git_manager.create_branch(branch_name)


def share_commit_with_agents(commit_info: Dict[str, Any], target_agents: List[str]) -> Dict[str, Any]:
    """Share commit information with other agents via A2A protocol"""
    try:
        from ..a2a.a2a_tools import send_message

        results = []
        for agent_id in target_agents:
            message_result = send_message(
                agent_id,
                "git_notification",
                f"New commit from {_git_manager.agent_id}: {commit_info.get('commit_message', 'No message')}",
                {
                    "commit_hash": commit_info.get("commit_hash"),
                    "branch": commit_info.get("branch", _git_manager.branch),
                    "files_changed": commit_info.get("files_changed", []),
                    "repo_url": _git_manager.repo_url,
                    "agent_role": _git_manager.project_role
                }
            )
            results.append({
                "agent_id": agent_id,
                "success": message_result["success"],
                "error": message_result.get("error")
            })

        return {
            "success": True,
            "shared_with": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "results": results
        }

    except Exception as e:
        logger.error(f"Failed to share commit with agents: {e}")
        return {
            "success": False,
            "error": str(e)
        }