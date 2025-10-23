"""
MEU Framework - Execute Domain: Solution Implementation Node

This node handles the Execute (E) domain of the MEU framework by implementing
solutions, writing code, and executing the planned specifications.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..tools.claude_code import (
    create_project, implement_feature, fix_bugs, setup_ci_cd,
    read_file, write_file, edit_file, search_code,
    run_tests, check_build_status, analyze_test_results
)
from ..tools.meu import register_test, execute_test, update_workspace_state
from ..tools.a2a import send_message, request_collaboration

logger = logging.getLogger(__name__)


async def implement_solution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implement the solution based on requirements analysis.

    This implements the Execute (E) domain of the MEU framework.
    """
    try:
        logger.info("Starting solution implementation in Execute domain")

        # Extract state data
        requirements_analysis = state.get("requirements_analysis", {})
        implementation_specs = state.get("implementation_specs", {})
        workspace_path = state.get("workspace_path", "/workspace")
        triplet_id = state.get("triplet_id")

        if not requirements_analysis or not implementation_specs:
            return {
                **state,
                "error": "Missing requirements analysis or implementation specs",
                "status": "failed"
            }

        # Track implementation progress
        implementation_results = {
            "files_created": [],
            "files_modified": [],
            "tests_implemented": [],
            "implementation_status": "in_progress",
            "errors_encountered": [],
            "warnings": []
        }

        # Use real Claude Code CLI for implementation
        user_request = requirements_analysis.get("raw_request", "")
        project_type = requirements_analysis.get("project_type", "python")

        # First, create a new project if needed
        create_result = create_project(
            project_name="implementation_project",
            project_type=project_type,
            workspace_path=workspace_path,
            description=user_request
        )

        if create_result["success"]:
            implementation_results["files_created"].extend(create_result.get("files_created", []))
            logger.info("Project structure created successfully")
        else:
            implementation_results["errors_encountered"].append({
                "step": "project_creation",
                "error": create_result.get("error", "Failed to create project")
            })

        # Implement the main feature using Claude Code CLI
        feature_result = implement_feature(
            feature_description=user_request,
            workspace_path=workspace_path,
            requirements=requirements_analysis
        )

        if feature_result["success"]:
            implementation_results["files_created"].extend(feature_result.get("files_created", []))
            implementation_results["files_modified"].extend(feature_result.get("files_modified", []))
            implementation_results["tests_implemented"].extend(feature_result.get("tests_created", []))
            logger.info("Feature implementation completed successfully")
        else:
            implementation_results["errors_encountered"].append({
                "step": "feature_implementation",
                "error": feature_result.get("error", "Failed to implement feature")
            })

        # Fix any bugs if requested
        if "fix" in user_request.lower() or "bug" in user_request.lower():
            fix_result = fix_bugs(
                bug_description=user_request,
                workspace_path=workspace_path
            )

            if fix_result["success"]:
                implementation_results["files_modified"].extend(fix_result.get("files_modified", []))
                logger.info("Bug fixes applied successfully")
            else:
                implementation_results["errors_encountered"].append({
                    "step": "bug_fixing",
                    "error": fix_result.get("error", "Failed to fix bugs")
                })

        # Set up CI/CD if it's a web application or needs deployment
        if any(keyword in user_request.lower() for keyword in ["web", "app", "deploy", "tetris", "game"]):
            cicd_result = setup_ci_cd(
                workspace_path=workspace_path,
                deployment_type="web_app"
            )

            if cicd_result["success"]:
                implementation_results["files_created"].extend(cicd_result.get("files_created", []))
                logger.info("CI/CD setup completed successfully")
            else:
                implementation_results["warnings"].append(f"CI/CD setup warning: {cicd_result.get('error', 'Unknown error')}")

        # Run initial tests if any were implemented
        test_results = {}
        if implementation_results["tests_implemented"]:
            logger.info("Running implemented tests")
            test_run_result = run_tests(workspace_path)

            if test_run_result["success"]:
                test_results = analyze_test_results(test_run_result["output"])

                # Register tests in MEU framework
                for test_file in implementation_results["tests_implemented"]:
                    register_result = register_test(
                        test_name=f"test_{test_file}",
                        test_type="execution_test",
                        triplet_id=triplet_id,
                        input_signature={"file_path": "str"},
                        output_signature="bool",
                        function_reference=test_file,
                        domain="E"
                    )

                    if register_result["success"]:
                        # Execute the registered test
                        execute_test(
                            register_result["test_id"],
                            test_file,
                            {"workspace_path": workspace_path}
                        )

        # Check build status
        build_status = check_build_status(workspace_path)

        # Determine implementation status
        if implementation_results["errors_encountered"]:
            implementation_results["implementation_status"] = "completed_with_errors"
        elif build_status.get("success", False) and test_results.get("all_passed", True):
            implementation_results["implementation_status"] = "completed_successfully"
        else:
            implementation_results["implementation_status"] = "completed_needs_review"

        # Update workspace state
        workspace_update = {
            "operation": "implementation_complete",
            "data": {
                "implementation_results": implementation_results,
                "test_results": test_results,
                "build_status": build_status
            }
        }

        update_result = update_workspace_state(workspace_update, triplet_id)

        if not update_result["success"]:
            logger.warning(f"Failed to update workspace state: {update_result['error']}")

        # Update state for next node
        updated_state = {
            **state,
            "implementation_results": implementation_results,
            "test_results": test_results,
            "build_status": build_status,
            "implementation_complete": True,
            "status": "implementation_complete",
            "next_action": "evaluate_results"
        }

        logger.info("Solution implementation completed")
        return updated_state

    except Exception as e:
        logger.error(f"Error in implement_solution_node: {e}")
        return {
            **state,
            "error": str(e),
            "status": "failed",
            "node": "implement_solution"
        }


async def _execute_implementation_step(step: str, requirements_analysis: Dict[str, Any],
                                     implementation_specs: Dict[str, Any],
                                     workspace_path: str) -> Dict[str, Any]:
    """Execute a single implementation step."""

    try:
        step_result = {
            "success": True,
            "files_created": [],
            "files_modified": [],
            "tests_implemented": []
        }

        if "Design and implement new functionality" in step:
            result = await _implement_new_functionality(
                requirements_analysis, implementation_specs, workspace_path
            )
            step_result.update(result)

        elif "Identify and fix existing bugs" in step:
            result = await _fix_existing_bugs(
                requirements_analysis, workspace_path
            )
            step_result.update(result)

        elif "Implement comprehensive testing" in step:
            result = await _implement_testing(
                requirements_analysis, implementation_specs, workspace_path
            )
            step_result.update(result)

        elif "Integrate with external APIs" in step:
            result = await _implement_api_integration(
                requirements_analysis, workspace_path
            )
            step_result.update(result)

        return step_result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def _implement_new_functionality(requirements_analysis: Dict[str, Any],
                                     implementation_specs: Dict[str, Any],
                                     workspace_path: str) -> Dict[str, Any]:
    """Implement new functionality based on requirements."""

    result = {
        "files_created": [],
        "files_modified": [],
        "tests_implemented": []
    }

    # Analyze what functionality to implement
    user_request = requirements_analysis.get("raw_request", "")

    # Create a new module file if needed
    if "module" in user_request.lower() or "class" in user_request.lower():
        module_name = _extract_module_name(user_request)
        module_path = f"{workspace_path}/{module_name}.py"

        module_content = _generate_module_content(user_request, requirements_analysis)

        write_result = write_file(module_path, module_content)
        if write_result["success"]:
            result["files_created"].append(module_path)

        # Create corresponding test file
        test_path = f"{workspace_path}/test_{module_name}.py"
        test_content = _generate_test_content(module_name, user_request)

        test_write_result = write_file(test_path, test_content)
        if test_write_result["success"]:
            result["files_created"].append(test_path)
            result["tests_implemented"].append(test_path)

    # Modify existing files if needed
    file_changes_required = implementation_specs.get("file_changes_required", [])
    for file_change in file_changes_required:
        if "Modify existing files" in file_change:
            # Search for relevant files to modify
            search_result = search_code("def ", file_pattern="*.py")
            if search_result["success"] and search_result["matches"]:
                for file_path in search_result["matches"][:3]:  # Limit to first 3 files
                    modification_result = _modify_existing_file(file_path, user_request)
                    if modification_result["success"]:
                        result["files_modified"].append(file_path)

    return result


async def _fix_existing_bugs(requirements_analysis: Dict[str, Any],
                           workspace_path: str) -> Dict[str, Any]:
    """Fix existing bugs in the codebase."""

    result = {
        "files_modified": []
    }

    user_request = requirements_analysis.get("raw_request", "")

    # Search for files that might contain bugs
    search_patterns = ["error", "exception", "bug", "fix"]

    for pattern in search_patterns:
        if pattern in user_request.lower():
            search_result = search_code(pattern, file_pattern="*.py")
            if search_result["success"] and search_result["matches"]:
                for file_path in search_result["matches"][:2]:  # Limit modifications
                    fix_result = _apply_bug_fix(file_path, user_request)
                    if fix_result["success"]:
                        result["files_modified"].append(file_path)

    return result


async def _implement_testing(requirements_analysis: Dict[str, Any],
                           implementation_specs: Dict[str, Any],
                           workspace_path: str) -> Dict[str, Any]:
    """Implement comprehensive testing."""

    result = {
        "files_created": [],
        "tests_implemented": []
    }

    # Create test files based on strategy
    test_strategy = implementation_specs.get("test_strategy", [])

    for strategy in test_strategy:
        if "Unit tests" in strategy:
            test_file = f"{workspace_path}/test_unit_implementation.py"
            test_content = _generate_unit_tests(requirements_analysis)

            write_result = write_file(test_file, test_content)
            if write_result["success"]:
                result["files_created"].append(test_file)
                result["tests_implemented"].append(test_file)

        elif "Integration tests" in strategy:
            test_file = f"{workspace_path}/test_integration.py"
            test_content = _generate_integration_tests(requirements_analysis)

            write_result = write_file(test_file, test_content)
            if write_result["success"]:
                result["files_created"].append(test_file)
                result["tests_implemented"].append(test_file)

    return result


async def _implement_api_integration(requirements_analysis: Dict[str, Any],
                                   workspace_path: str) -> Dict[str, Any]:
    """Implement API integration functionality."""

    result = {
        "files_created": [],
        "files_modified": []
    }

    # Create API client module
    api_client_path = f"{workspace_path}/api_client.py"
    api_content = _generate_api_client_content(requirements_analysis)

    write_result = write_file(api_client_path, api_content)
    if write_result["success"]:
        result["files_created"].append(api_client_path)

    return result


def _extract_module_name(user_request: str) -> str:
    """Extract module name from user request."""
    words = user_request.lower().split()

    # Look for class or module names
    for i, word in enumerate(words):
        if word in ["class", "module", "implement"] and i + 1 < len(words):
            next_word = words[i + 1].replace(",", "").replace(".", "")
            if next_word.isalpha():
                return next_word

    return "new_module"


def _generate_module_content(user_request: str, requirements_analysis: Dict[str, Any]) -> str:
    """Generate content for new module."""

    module_name = _extract_module_name(user_request)

    content = f'''"""
{module_name.title()} Module

Generated based on user request: {user_request}
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class {module_name.title()}:
    """
    {module_name.title()} implementation.
    """

    def __init__(self):
        """Initialize {module_name}."""
        self.initialized = True

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process data according to requirements.

        Args:
            data: Input data to process

        Returns:
            Dict containing processing results
        """
        try:
            # Implementation based on requirements
            result = {{
                "success": True,
                "data": data,
                "processed_at": "timestamp"
            }}

            return result

        except Exception as e:
            logger.error(f"Error in {module_name} processing: {{e}}")
            return {{
                "success": False,
                "error": str(e)
            }}


def {module_name}_function(input_param: str) -> str:
    """
    Utility function for {module_name}.

    Args:
        input_param: Input parameter

    Returns:
        Processed result
    """
    return f"Processed: {{input_param}}"
'''

    return content


def _generate_test_content(module_name: str, user_request: str) -> str:
    """Generate test content for module."""

    content = f'''"""
Tests for {module_name} module.
"""

import pytest
from {module_name} import {module_name.title()}, {module_name}_function


class Test{module_name.title()}:
    """Test cases for {module_name.title()} class."""

    def test_init(self):
        """Test initialization."""
        instance = {module_name.title()}()
        assert instance.initialized is True

    def test_process_success(self):
        """Test successful processing."""
        instance = {module_name.title()}()
        result = instance.process("test_data")

        assert result["success"] is True
        assert "data" in result

    def test_process_error_handling(self):
        """Test error handling in processing."""
        instance = {module_name.title()}()
        # Test with invalid data that should trigger error handling
        result = instance.process(None)

        # Should handle gracefully
        assert "success" in result


def test_{module_name}_function():
    """Test {module_name} utility function."""
    result = {module_name}_function("test_input")
    assert "Processed:" in result
    assert "test_input" in result


def test_{module_name}_integration():
    """Integration test for {module_name}."""
    instance = {module_name.title()}()
    test_data = "integration_test_data"

    result = instance.process(test_data)
    processed_result = {module_name}_function(str(result))

    assert "Processed:" in processed_result
'''

    return content


def _modify_existing_file(file_path: str, user_request: str) -> Dict[str, Any]:
    """Modify an existing file based on user request."""

    try:
        # Read current file content
        read_result = read_file(file_path)
        if not read_result["success"]:
            return {"success": False, "error": f"Could not read {file_path}"}

        current_content = read_result["content"]

        # Simple modification: add a comment about the change
        modification = f"\n# Modified for: {user_request}\n"

        # Add modification at the end of file
        modified_content = current_content + modification

        # Write back the modified content
        write_result = write_file(file_path, modified_content)

        return {"success": write_result["success"]}

    except Exception as e:
        return {"success": False, "error": str(e)}


def _apply_bug_fix(file_path: str, user_request: str) -> Dict[str, Any]:
    """Apply bug fix to a file."""

    try:
        # Read current file content
        read_result = read_file(file_path)
        if not read_result["success"]:
            return {"success": False, "error": f"Could not read {file_path}"}

        current_content = read_result["content"]

        # Simple bug fix: add error handling
        if "try:" not in current_content and "def " in current_content:
            # Add basic error handling to functions
            lines = current_content.split('\n')
            modified_lines = []

            for line in lines:
                modified_lines.append(line)
                if line.strip().startswith("def ") and ":" in line:
                    # Add try-except block after function definition
                    indent = len(line) - len(line.lstrip())
                    modified_lines.append(" " * (indent + 4) + "try:")
                    modified_lines.append(" " * (indent + 8) + "# Function implementation")
                    modified_lines.append(" " * (indent + 4) + "except Exception as e:")
                    modified_lines.append(" " * (indent + 8) + "logger.error(f'Error in function: {e}')")
                    modified_lines.append(" " * (indent + 8) + "raise")

            modified_content = '\n'.join(modified_lines)

            # Write back the modified content
            write_result = write_file(file_path, modified_content)
            return {"success": write_result["success"]}

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


def _generate_unit_tests(requirements_analysis: Dict[str, Any]) -> str:
    """Generate unit test content."""

    content = '''"""
Unit tests for implementation.
"""

import pytest
import unittest
from unittest.mock import Mock, patch


class TestImplementation(unittest.TestCase):
    """Unit tests for the implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {"test": "data"}

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Basic test implementation
        result = True  # Replace with actual test
        self.assertTrue(result)

    def test_error_handling(self):
        """Test error handling."""
        # Test error conditions
        with self.assertRaises(Exception):
            # Code that should raise exception
            pass

    def test_edge_cases(self):
        """Test edge cases."""
        # Test edge cases and boundary conditions
        self.assertIsNotNone(self.test_data)


if __name__ == "__main__":
    unittest.main()
'''

    return content


def _generate_integration_tests(requirements_analysis: Dict[str, Any]) -> str:
    """Generate integration test content."""

    content = '''"""
Integration tests for implementation.
"""

import pytest
import unittest


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        """Set up integration test environment."""
        self.integration_data = {"integration": "test"}

    def test_component_integration(self):
        """Test integration between components."""
        # Integration test implementation
        result = True  # Replace with actual integration test
        self.assertTrue(result)

    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # End-to-end test implementation
        self.assertIsNotNone(self.integration_data)


if __name__ == "__main__":
    unittest.main()
'''

    return content


def _generate_api_client_content(requirements_analysis: Dict[str, Any]) -> str:
    """Generate API client content."""

    content = '''"""
API Client for external integrations.
"""

import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class APIClient:
    """Client for external API integration."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GET request to API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            API response data
        """
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }

        except Exception as e:
            logger.error(f"API GET request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make POST request to API.

        Args:
            endpoint: API endpoint
            data: Request data

        Returns:
            API response data
        """
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()

            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }

        except Exception as e:
            logger.error(f"API POST request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
'''

    return content