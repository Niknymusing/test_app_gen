"""
MEU Framework - Model Domain: Requirements Analysis Node

This node handles the Model (M) domain of the MEU framework by analyzing
requirements, specifications, and creating detailed implementation plans.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..tools.claude_code import read_file, search_code, analyze_test_results
from ..tools.meu import create_triplet, update_workspace_state
from ..tools.mcp import get_resources, inject_context

logger = logging.getLogger(__name__)


async def analyze_requirements_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze requirements and create detailed implementation specifications.

    This implements the Model (M) domain of the MEU framework.
    """
    try:
        logger.info("Starting requirements analysis in Model domain")

        # Extract input data
        user_request = state.get("user_request", "")
        workspace_path = state.get("workspace_path", "/workspace")
        triplet_id = state.get("triplet_id")

        # Create analysis triplet if not exists
        if not triplet_id:
            triplet_result = create_triplet(
                triplet_type="source",
                model_spec={
                    "specification_type": "requirements_analysis",
                    "requirements": [user_request],
                    "constraints": ["use_claude_code_tools", "meu_framework_compliance"]
                },
                execute_env={
                    "environment_type": "local",
                    "workspace_path": workspace_path,
                    "test_framework": "pytest"
                },
                update_criteria={
                    "success_criteria": ["clear_requirements", "testable_specifications"],
                    "evaluation_metrics": ["completeness", "clarity", "feasibility"]
                }
            )

            if triplet_result["success"]:
                triplet_id = triplet_result["triplet_id"]
                state["triplet_id"] = triplet_id
            else:
                return {
                    **state,
                    "error": f"Failed to create analysis triplet: {triplet_result['error']}",
                    "status": "failed"
                }

        # Analyze existing codebase context
        codebase_analysis = await _analyze_existing_codebase(workspace_path)

        # Get relevant resources through MCP
        mcp_resources = get_resources(resource_type="file", file_extensions=[".py", ".md", ".yaml", ".json"])
        relevant_context = {}

        if mcp_resources["success"]:
            # Inject context from relevant files
            resource_ids = [r["id"] for r in mcp_resources["resources"][:10]]  # Limit to first 10
            context_result = inject_context(resource_ids, "requirements_analysis")
            if context_result["success"]:
                relevant_context = context_result["enhanced_context"]

        # Analyze requirements
        requirements_analysis = _analyze_requirements(
            user_request,
            codebase_analysis,
            relevant_context
        )

        # Create detailed specifications
        implementation_specs = _create_implementation_specs(requirements_analysis)

        # Update workspace state
        workspace_update = {
            "operation": "requirements_analysis_complete",
            "data": {
                "requirements_analysis": requirements_analysis,
                "implementation_specs": implementation_specs,
                "codebase_context": codebase_analysis,
                "mcp_context": relevant_context
            }
        }

        update_result = update_workspace_state(workspace_update, triplet_id)

        if not update_result["success"]:
            logger.warning(f"Failed to update workspace state: {update_result['error']}")

        # Update state for next node
        updated_state = {
            **state,
            "requirements_analysis": requirements_analysis,
            "implementation_specs": implementation_specs,
            "codebase_context": codebase_analysis,
            "mcp_context": relevant_context,
            "analysis_complete": True,
            "status": "analysis_complete",
            "next_action": "implement_solution"
        }

        logger.info("Requirements analysis completed successfully")
        return updated_state

    except Exception as e:
        logger.error(f"Error in analyze_requirements_node: {e}")
        return {
            **state,
            "error": str(e),
            "status": "failed",
            "node": "analyze_requirements"
        }


async def _analyze_existing_codebase(workspace_path: str) -> Dict[str, Any]:
    """Analyze existing codebase structure and content."""
    try:
        analysis = {
            "files_analyzed": [],
            "project_structure": {},
            "dependencies": [],
            "test_files": [],
            "config_files": [],
            "main_modules": []
        }

        # Search for key files
        search_patterns = [
            ("requirements.txt", "dependencies"),
            ("package.json", "dependencies"),
            ("test_*.py", "test_files"),
            ("*_test.py", "test_files"),
            ("config*.py", "config_files"),
            ("settings*.py", "config_files"),
            ("main.py", "main_modules"),
            ("app.py", "main_modules")
        ]

        for pattern, category in search_patterns:
            search_result = search_code(pattern, file_pattern="*")
            if search_result["success"]:
                for file_path in search_result["matches"]:
                    analysis[category].append(file_path)

                    # Read file content for analysis
                    file_content = read_file(file_path)
                    if file_content["success"]:
                        analysis["files_analyzed"].append({
                            "path": file_path,
                            "size": len(file_content["content"]),
                            "lines": len(file_content["content"].splitlines())
                        })

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing codebase: {e}")
        return {"error": str(e)}


def _analyze_requirements(user_request: str, codebase_analysis: Dict[str, Any],
                         mcp_context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze and structure the user requirements."""

    analysis = {
        "raw_request": user_request,
        "parsed_requirements": [],
        "technical_constraints": [],
        "functional_requirements": [],
        "non_functional_requirements": [],
        "dependencies": [],
        "complexity_assessment": "medium",
        "estimated_effort": "moderate"
    }

    # Parse requirements from user request
    request_lower = user_request.lower()

    # Detect project type
    project_type = "python"  # default
    if any(keyword in request_lower for keyword in ["web", "html", "css", "javascript", "react", "node"]):
        project_type = "web"
    elif any(keyword in request_lower for keyword in ["game", "tetris", "browser app"]):
        project_type = "web_game"
    elif any(keyword in request_lower for keyword in ["api", "server", "backend"]):
        project_type = "api"
    elif any(keyword in request_lower for keyword in ["cli", "command line", "script"]):
        project_type = "cli"

    analysis["project_type"] = project_type

    # Identify functional requirements
    if "implement" in request_lower or "create" in request_lower or "build" in request_lower:
        analysis["functional_requirements"].append("New implementation required")
    if "fix" in request_lower or "bug" in request_lower:
        analysis["functional_requirements"].append("Bug fix required")
    if "test" in request_lower:
        analysis["functional_requirements"].append("Testing component required")
    if "api" in request_lower:
        analysis["functional_requirements"].append("API integration required")
    if "deploy" in request_lower:
        analysis["functional_requirements"].append("Deployment setup required")

    # Identify technical constraints
    if "fast" in request_lower or "performance" in request_lower:
        analysis["technical_constraints"].append("Performance optimization required")
    if "secure" in request_lower or "security" in request_lower:
        analysis["technical_constraints"].append("Security considerations required")
    if "scale" in request_lower:
        analysis["technical_constraints"].append("Scalability considerations required")

    # Assess complexity based on codebase and requirements
    complexity_indicators = [
        len(codebase_analysis.get("files_analyzed", [])),
        len(analysis["functional_requirements"]),
        len(analysis["technical_constraints"])
    ]

    total_complexity = sum(complexity_indicators)
    if total_complexity > 10:
        analysis["complexity_assessment"] = "high"
        analysis["estimated_effort"] = "significant"
    elif total_complexity > 5:
        analysis["complexity_assessment"] = "medium"
        analysis["estimated_effort"] = "moderate"
    else:
        analysis["complexity_assessment"] = "low"
        analysis["estimated_effort"] = "minimal"

    # Extract dependencies from MCP context
    if mcp_context and "resources" in mcp_context:
        for resource_id, resource_data in mcp_context["resources"].items():
            if resource_data.get("type") == "code":
                analysis["dependencies"].append(resource_id)

    analysis["parsed_requirements"] = [
        user_request,
        f"Complexity: {analysis['complexity_assessment']}",
        f"Estimated effort: {analysis['estimated_effort']}"
    ]

    return analysis


def _create_implementation_specs(requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create detailed implementation specifications."""

    specs = {
        "implementation_plan": [],
        "file_changes_required": [],
        "new_files_needed": [],
        "test_strategy": [],
        "validation_criteria": [],
        "risk_assessment": [],
        "timeline_estimate": "TBD"
    }

    # Create implementation plan based on requirements
    functional_reqs = requirements_analysis.get("functional_requirements", [])

    for req in functional_reqs:
        if "New implementation" in req:
            specs["implementation_plan"].append("Design and implement new functionality")
            specs["new_files_needed"].append("New module files")
            specs["test_strategy"].append("Unit tests for new functionality")

        if "Bug fix" in req:
            specs["implementation_plan"].append("Identify and fix existing bugs")
            specs["file_changes_required"].append("Modify existing files")
            specs["test_strategy"].append("Regression tests")

        if "Testing component" in req:
            specs["implementation_plan"].append("Implement comprehensive testing")
            specs["new_files_needed"].append("Test files")
            specs["test_strategy"].append("Automated test suite")

        if "API integration" in req:
            specs["implementation_plan"].append("Integrate with external APIs")
            specs["file_changes_required"].append("API client modules")
            specs["test_strategy"].append("API integration tests")

    # Set validation criteria
    specs["validation_criteria"] = [
        "All tests pass",
        "Code follows project conventions",
        "Requirements fully implemented",
        "No new security vulnerabilities"
    ]

    # Assess risks
    complexity = requirements_analysis.get("complexity_assessment", "medium")
    if complexity == "high":
        specs["risk_assessment"] = [
            "High complexity may lead to longer development time",
            "Multiple components may need coordination",
            "Testing may be complex"
        ]
    elif complexity == "medium":
        specs["risk_assessment"] = [
            "Moderate complexity manageable with careful planning",
            "Standard testing approach should suffice"
        ]
    else:
        specs["risk_assessment"] = [
            "Low complexity, minimal risks expected"
        ]

    # Estimate timeline based on complexity
    effort = requirements_analysis.get("estimated_effort", "moderate")
    timeline_map = {
        "minimal": "1-2 hours",
        "moderate": "3-6 hours",
        "significant": "1-2 days"
    }
    specs["timeline_estimate"] = timeline_map.get(effort, "TBD")

    return specs