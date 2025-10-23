"""
Claude Code Integration Tools for MEU Framework

This module provides advanced Claude Code functionality integration
for the MEU framework with enhanced error handling, workspace management,
and integration with A2A and MCP protocols.
"""

import subprocess
import json
import os
import logging
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


class ClaudeCodeIntegration:
    """Advanced Claude Code integration for MEU framework"""

    def __init__(self, workspace_path: str = ".", docker_env: Optional[Dict[str, Any]] = None):
        self.workspace_path = Path(workspace_path)
        self.docker_env = docker_env or {}
        self._setup_workspace()

    def _setup_workspace(self):
        """Setup the workspace for Claude Code operations"""
        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created workspace directory: {self.workspace_path}")

    def _execute_in_environment(self, command: List[str], working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute command in the configured environment (local or Docker)"""
        if self.docker_env.get("enabled", False):
            return self._execute_in_docker(command, working_dir)
        else:
            return self._execute_locally(command, working_dir)

    def _execute_locally(self, command: List[str], working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute command locally"""
        try:
            work_dir = working_dir or str(self.workspace_path)
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(command)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "command": " ".join(command)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(command)
            }

    def _execute_in_docker(self, command: List[str], working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute command in Docker environment with non-root user for Claude Code"""
        try:
            container_name = self.docker_env.get("container_name", "meu-coder-env")

            # Check if this is a Claude Code command that needs non-root execution
            is_claude_command = any("claude" in str(cmd).lower() for cmd in command)

            if is_claude_command:
                # Use non-root user execution for Claude Code to enable --dangerously-skip-permissions
                work_dir = working_dir or "/workspace"
                docker_command = [
                    "docker", "exec", "-w", work_dir, "-u", "meuuser", container_name
                ] + command
            else:
                # Regular command execution
                docker_command = [
                    "docker", "exec", container_name
                ] + command

            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(command),
                "environment": "docker",
                "execution_method": "non_root_user" if is_claude_command else "standard"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(command),
                "environment": "docker"
            }


# Global instance
_claude_code = ClaudeCodeIntegration()


def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read a file using Claude Code's enhanced read capabilities

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        Dict containing file contents and metadata
    """
    try:
        full_path = _claude_code.workspace_path / file_path

        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_path": str(full_path)
            }

        with open(full_path, 'r', encoding=encoding) as f:
            content = f.read()

        # Enhanced metadata
        stat = full_path.stat()
        lines = content.splitlines()

        return {
            "success": True,
            "content": content,
            "file_path": str(full_path),
            "relative_path": file_path,
            "size": len(content),
            "size_bytes": stat.st_size,
            "lines": len(lines),
            "encoding": encoding,
            "is_binary": False,
            "last_modified": stat.st_mtime,
            "file_type": full_path.suffix,
            "metadata": {
                "blank_lines": sum(1 for line in lines if not line.strip()),
                "non_blank_lines": sum(1 for line in lines if line.strip())
            }
        }
    except UnicodeDecodeError:
        # Handle binary files
        return {
            "success": False,
            "error": f"File appears to be binary or uses unsupported encoding: {file_path}",
            "file_path": str(full_path),
            "is_binary": True
        }
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": str(full_path)
        }


def write_file(file_path: str, content: str, create_dirs: bool = True, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Write content to a file with enhanced error handling

    Args:
        file_path: Path where to write the file
        content: Content to write
        create_dirs: Whether to create parent directories
        encoding: File encoding

    Returns:
        Dict containing operation result
    """
    try:
        full_path = _claude_code.workspace_path / file_path

        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file if it exists
        backup_path = None
        if full_path.exists():
            backup_path = full_path.with_suffix(f"{full_path.suffix}.backup")
            shutil.copy2(full_path, backup_path)

        with open(full_path, 'w', encoding=encoding) as f:
            f.write(content)

        # Validate the write
        with open(full_path, 'r', encoding=encoding) as f:
            written_content = f.read()

        if written_content != content:
            raise ValueError("File write validation failed")

        return {
            "success": True,
            "file_path": str(full_path),
            "relative_path": file_path,
            "bytes_written": len(content.encode(encoding)),
            "lines_written": len(content.splitlines()),
            "encoding": encoding,
            "backup_created": backup_path is not None,
            "backup_path": str(backup_path) if backup_path else None
        }
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": str(full_path) if 'full_path' in locals() else file_path
        }


def edit_file(file_path: str, old_content: str, new_content: str, backup: bool = True) -> Dict[str, Any]:
    """
    Edit a file by replacing old content with new content

    Args:
        file_path: Path to the file to edit
        old_content: Content to replace
        new_content: New content to insert
        backup: Whether to create a backup

    Returns:
        Dict containing operation result
    """
    try:
        full_path = _claude_code.workspace_path / file_path

        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_path": str(full_path)
            }

        with open(full_path, 'r', encoding='utf-8') as f:
            current_content = f.read()

        if old_content not in current_content:
            return {
                "success": False,
                "error": "Old content not found in file",
                "file_path": str(full_path),
                "suggestion": "Use search_code to find the exact content to replace"
            }

        # Create backup if requested
        backup_path = None
        if backup:
            backup_path = full_path.with_suffix(f"{full_path.suffix}.backup")
            shutil.copy2(full_path, backup_path)

        # Perform replacement
        updated_content = current_content.replace(old_content, new_content)
        replacements_made = current_content.count(old_content)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        return {
            "success": True,
            "file_path": str(full_path),
            "relative_path": file_path,
            "replacements_made": replacements_made,
            "old_size": len(current_content),
            "new_size": len(updated_content),
            "backup_created": backup_path is not None,
            "backup_path": str(backup_path) if backup_path else None
        }
    except Exception as e:
        logger.error(f"Error editing file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": str(full_path) if 'full_path' in locals() else file_path
        }


def search_code(pattern: str, directory: str = ".", file_extensions: Optional[List[str]] = None,
                max_results: int = 100, context_lines: int = 2) -> Dict[str, Any]:
    """
    Search for code patterns with enhanced capabilities

    Args:
        pattern: Search pattern (supports regex)
        directory: Directory to search in
        file_extensions: List of file extensions to search
        max_results: Maximum number of results to return
        context_lines: Number of context lines around matches

    Returns:
        Dict containing search results
    """
    try:
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php']

        search_dir = _claude_code.workspace_path / directory

        if not search_dir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }

        results = []
        file_count = 0

        # Compile regex pattern
        try:
            regex_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return {
                "success": False,
                "error": f"Invalid regex pattern: {e}"
            }

        for ext in file_extensions:
            for file_path in search_dir.rglob(f"*{ext}"):
                if len(results) >= max_results:
                    break

                file_count += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()

                    # Search for matches
                    for match in regex_pattern.finditer(content):
                        if len(results) >= max_results:
                            break

                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1

                        # Get context
                        start_line = max(0, line_num - context_lines - 1)
                        end_line = min(len(lines), line_num + context_lines)
                        context = lines[start_line:end_line]

                        results.append({
                            "file": str(file_path.relative_to(_claude_code.workspace_path)),
                            "line": line_num,
                            "column": match.start() - content.rfind('\n', 0, match.start()),
                            "match": match.group(),
                            "context": context,
                            "pattern": pattern
                        })

                except (UnicodeDecodeError, PermissionError):
                    continue
                except Exception as e:
                    logger.warning(f"Error searching in {file_path}: {e}")
                    continue

        return {
            "success": True,
            "pattern": pattern,
            "results": results,
            "total_matches": len(results),
            "files_searched": file_count,
            "truncated": len(results) >= max_results
        }
    except Exception as e:
        logger.error(f"Error searching for pattern '{pattern}': {e}")
        return {
            "success": False,
            "error": str(e),
            "pattern": pattern
        }


def run_tests(test_command: str = "pytest", working_directory: str = ".",
              timeout: int = 300, environment_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Run tests with enhanced environment and result analysis

    Args:
        test_command: Command to run tests
        working_directory: Directory to run tests in
        timeout: Timeout in seconds
        environment_vars: Additional environment variables

    Returns:
        Dict containing test results and analysis
    """
    try:
        # Setup environment
        env = os.environ.copy()
        if environment_vars:
            env.update(environment_vars)

        work_dir = _claude_code.workspace_path / working_directory

        # Execute tests
        result = subprocess.run(
            test_command.split(),
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        # Analyze results
        analysis = analyze_test_results(result.stdout + result.stderr)

        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": test_command,
            "working_directory": str(work_dir),
            "analysis": analysis.get("analysis", {}),
            "environment": "local"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Test execution timed out ({timeout} seconds)",
            "command": test_command
        }
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return {
            "success": False,
            "error": str(e),
            "command": test_command
        }


def check_build_status(build_command: Optional[str] = None, target_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Check build status with enhanced analysis

    Args:
        build_command: Custom build command
        target_files: Specific files to check

    Returns:
        Dict containing build status and details
    """
    try:
        if build_command:
            # Run custom build command
            result = _claude_code._execute_locally(build_command.split())
            return {
                "success": result["success"],
                "build_command": build_command,
                "output": result.get("stdout", ""),
                "errors": result.get("stderr", ""),
                "custom_build": True
            }

        # Default Python compilation check
        if target_files is None:
            target_files = list(_claude_code.workspace_path.rglob("*.py"))
            target_files = [str(f.relative_to(_claude_code.workspace_path)) for f in target_files
                           if not str(f).startswith('.') and '__pycache__' not in str(f)]

        build_results = []
        overall_success = True

        for file_path in target_files:
            try:
                full_path = _claude_code.workspace_path / file_path
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(full_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                file_success = result.returncode == 0
                overall_success = overall_success and file_success

                build_results.append({
                    "file": file_path,
                    "success": file_success,
                    "error": result.stderr if not file_success else None,
                    "type": "syntax_check"
                })
            except Exception as e:
                overall_success = False
                build_results.append({
                    "file": file_path,
                    "success": False,
                    "error": str(e),
                    "type": "syntax_check"
                })

        return {
            "success": overall_success,
            "build_command": "python -m py_compile",
            "files_checked": len(target_files),
            "files_passed": sum(1 for r in build_results if r["success"]),
            "files_failed": sum(1 for r in build_results if not r["success"]),
            "results": build_results,
            "custom_build": False
        }
    except Exception as e:
        logger.error(f"Error checking build status: {e}")
        return {
            "success": False,
            "error": str(e),
            "build_command": build_command or "python -m py_compile"
        }


def analyze_test_results(test_output: str) -> Dict[str, Any]:
    """
    Analyze test results with enhanced parsing and metrics

    Args:
        test_output: Raw output from test execution

    Returns:
        Dict containing analyzed test metrics
    """
    try:
        analysis = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "success_rate": 0.0,
            "failure_details": [],
            "error_details": [],
            "summary": "",
            "test_framework": "unknown",
            "execution_time": None,
            "coverage": None
        }

        lines = test_output.splitlines()

        # Detect test framework
        if "pytest" in test_output.lower():
            analysis["test_framework"] = "pytest"
            analysis.update(_parse_pytest_output(lines))
        elif "unittest" in test_output.lower():
            analysis["test_framework"] = "unittest"
            analysis.update(_parse_unittest_output(lines))
        else:
            analysis.update(_parse_generic_output(lines))

        # Calculate totals and success rate
        analysis["total_tests"] = (analysis["passed"] + analysis["failed"] +
                                 analysis["errors"] + analysis["skipped"])

        if analysis["total_tests"] > 0:
            analysis["success_rate"] = analysis["passed"] / analysis["total_tests"]

        # Create summary
        if analysis["total_tests"] > 0:
            analysis["summary"] = (f"Tests: {analysis['total_tests']}, "
                                 f"Passed: {analysis['passed']}, "
                                 f"Failed: {analysis['failed']}, "
                                 f"Errors: {analysis['errors']}, "
                                 f"Skipped: {analysis['skipped']}, "
                                 f"Success Rate: {analysis['success_rate']:.1%}")
        else:
            analysis["summary"] = "No test results found in output"

        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing test results: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _parse_pytest_output(lines: List[str]) -> Dict[str, Any]:
    """Parse pytest-specific output"""
    result = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    for line in lines:
        line = line.strip().lower()
        # Look for pytest summary line
        if " passed" in line or " failed" in line:
            # Extract numbers using regex
            import re
            numbers = re.findall(r'(\d+)\s+(passed|failed|error|skipped)', line)
            for num, status in numbers:
                if status in result:
                    result[status] = int(num)
                elif status == "error":
                    result["errors"] = int(num)

    return result


def _parse_unittest_output(lines: List[str]) -> Dict[str, Any]:
    """Parse unittest-specific output"""
    result = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    for line in lines:
        if "Ran " in line and " test" in line:
            import re
            match = re.search(r'Ran (\d+) test', line)
            if match:
                total = int(match.group(1))
                result["passed"] = total  # Default assumption

        if "FAILED" in line:
            import re
            match = re.search(r'failures=(\d+)', line)
            if match:
                result["failed"] = int(match.group(1))
                result["passed"] -= result["failed"]

    return result


def _parse_generic_output(lines: List[str]) -> Dict[str, Any]:
    """Parse generic test output"""
    result = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    for line in lines:
        line_lower = line.lower()
        if "pass" in line_lower:
            result["passed"] += line_lower.count("pass")
        if "fail" in line_lower:
            result["failed"] += line_lower.count("fail")
        if "error" in line_lower:
            result["errors"] += line_lower.count("error")
        if "skip" in line_lower:
            result["skipped"] += line_lower.count("skip")

    return result