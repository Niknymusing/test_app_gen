"""
MEU Framework - Update Domain: Results Evaluation Node

This node handles the Update (U) domain of the MEU framework by evaluating
implementation results, providing feedback, and determining next actions.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..tools.claude_code import run_tests, check_build_status, analyze_test_results
from ..tools.meu import evaluate_triplet, update_workspace_state
from ..tools.a2a import send_message, share_task_result
from ..tools.mcp import sync_workspace_state

logger = logging.getLogger(__name__)


async def evaluate_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate implementation results and provide feedback.

    This implements the Update (U) domain of the MEU framework.
    """
    try:
        logger.info("Starting results evaluation in Update domain")

        # Extract state data
        requirements_analysis = state.get("requirements_analysis", {})
        implementation_results = state.get("implementation_results", {})
        test_results = state.get("test_results", {})
        build_status = state.get("build_status", {})
        workspace_path = state.get("workspace_path", "/workspace")
        triplet_id = state.get("triplet_id")

        if not implementation_results:
            return {
                **state,
                "error": "Missing implementation results for evaluation",
                "status": "failed"
            }

        # Perform comprehensive evaluation
        evaluation_results = await _perform_comprehensive_evaluation(
            requirements_analysis,
            implementation_results,
            test_results,
            build_status,
            workspace_path
        )

        # Evaluate the MEU triplet
        triplet_evaluation = None
        if triplet_id:
            evaluation_criteria = {
                "completion_requirements": requirements_analysis.get("functional_requirements", []),
                "quality_metrics": ["test_coverage", "build_success", "error_rate"],
                "performance_criteria": ["execution_time", "resource_usage"],
                "user_satisfaction": ["requirements_met", "usability"]
            }

            triplet_eval_result = evaluate_triplet(triplet_id, evaluation_criteria)
            if triplet_eval_result["success"]:
                triplet_evaluation = triplet_eval_result["evaluation"]

        # Generate recommendations
        recommendations = _generate_recommendations(
            evaluation_results,
            requirements_analysis,
            implementation_results
        )

        # Determine final status and next actions
        final_status = _determine_final_status(evaluation_results)
        next_actions = _determine_next_actions(evaluation_results, recommendations)

        # Create evaluation summary
        evaluation_summary = {
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "overall_assessment": evaluation_results,
            "triplet_evaluation": triplet_evaluation,
            "recommendations": recommendations,
            "final_status": final_status,
            "next_actions": next_actions,
            "success_rate": evaluation_results.get("overall_score", 0.0),
            "areas_for_improvement": recommendations.get("improvements", []),
            "completed_successfully": final_status in ["completed", "completed_with_minor_issues"]
        }

        # Update workspace state with final evaluation
        workspace_update = {
            "operation": "evaluation_complete",
            "data": {
                "evaluation_summary": evaluation_summary,
                "final_implementation_status": final_status,
                "recommendations": recommendations
            }
        }

        update_result = update_workspace_state(workspace_update, triplet_id)

        if not update_result["success"]:
            logger.warning(f"Failed to update workspace state: {update_result['error']}")

        # Sync workspace state via MCP if needed
        if triplet_id:
            sync_result = sync_workspace_state(triplet_id, workspace_update)
            if sync_result["success"]:
                logger.info("Workspace state synchronized via MCP")

        # Share results via A2A if collaboration was involved
        if state.get("collaboration_agents"):
            for agent_id in state["collaboration_agents"]:
                share_result = share_task_result(
                    agent_id,
                    triplet_id or "unknown",
                    {
                        "evaluation_summary": evaluation_summary,
                        "implementation_results": implementation_results,
                        "artifacts": implementation_results.get("files_created", [])
                    },
                    workspace_update
                )

                if share_result["success"]:
                    logger.info(f"Results shared with agent {agent_id}")

        # Final state update
        updated_state = {
            **state,
            "evaluation_summary": evaluation_summary,
            "final_status": final_status,
            "recommendations": recommendations,
            "next_actions": next_actions,
            "evaluation_complete": True,
            "status": final_status,
            "completed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Results evaluation completed with status: {final_status}")
        return updated_state

    except Exception as e:
        logger.error(f"Error in evaluate_results_node: {e}")
        return {
            **state,
            "error": str(e),
            "status": "failed",
            "node": "evaluate_results"
        }


async def _perform_comprehensive_evaluation(requirements_analysis: Dict[str, Any],
                                          implementation_results: Dict[str, Any],
                                          test_results: Dict[str, Any],
                                          build_status: Dict[str, Any],
                                          workspace_path: str) -> Dict[str, Any]:
    """Perform comprehensive evaluation of implementation results."""

    evaluation = {
        "requirement_fulfillment": 0.0,
        "implementation_quality": 0.0,
        "test_coverage": 0.0,
        "build_success": 0.0,
        "error_rate": 0.0,
        "overall_score": 0.0,
        "detailed_scores": {},
        "issues_found": [],
        "strengths": []
    }

    # Evaluate requirement fulfillment
    requirement_score = _evaluate_requirement_fulfillment(
        requirements_analysis, implementation_results
    )
    evaluation["requirement_fulfillment"] = requirement_score["score"]
    evaluation["detailed_scores"]["requirements"] = requirement_score

    # Evaluate implementation quality
    quality_score = _evaluate_implementation_quality(implementation_results)
    evaluation["implementation_quality"] = quality_score["score"]
    evaluation["detailed_scores"]["quality"] = quality_score

    # Evaluate test coverage and results
    test_score = _evaluate_test_results(test_results, implementation_results)
    evaluation["test_coverage"] = test_score["score"]
    evaluation["detailed_scores"]["testing"] = test_score

    # Evaluate build status
    build_score = _evaluate_build_status(build_status)
    evaluation["build_success"] = build_score["score"]
    evaluation["detailed_scores"]["build"] = build_score

    # Calculate error rate
    error_score = _evaluate_error_rate(implementation_results)
    evaluation["error_rate"] = error_score["score"]
    evaluation["detailed_scores"]["errors"] = error_score

    # Run additional tests if needed
    if workspace_path:
        additional_test_result = run_tests(workspace_path)
        if additional_test_result["success"]:
            additional_analysis = analyze_test_results(additional_test_result["output"])
            evaluation["additional_test_results"] = additional_analysis

    # Calculate overall score
    scores = [
        evaluation["requirement_fulfillment"],
        evaluation["implementation_quality"],
        evaluation["test_coverage"],
        evaluation["build_success"],
        (1.0 - evaluation["error_rate"])  # Invert error rate for scoring
    ]

    evaluation["overall_score"] = sum(scores) / len(scores)

    # Collect issues and strengths
    for score_category, score_data in evaluation["detailed_scores"].items():
        if score_data.get("issues"):
            evaluation["issues_found"].extend(score_data["issues"])
        if score_data.get("strengths"):
            evaluation["strengths"].extend(score_data["strengths"])

    return evaluation


def _evaluate_requirement_fulfillment(requirements_analysis: Dict[str, Any],
                                     implementation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate how well requirements were fulfilled."""

    functional_reqs = requirements_analysis.get("functional_requirements", [])
    files_created = implementation_results.get("files_created", [])
    files_modified = implementation_results.get("files_modified", [])

    score_data = {
        "score": 0.0,
        "details": {},
        "issues": [],
        "strengths": []
    }

    if not functional_reqs:
        score_data["score"] = 0.8  # Default score if no specific requirements
        score_data["details"]["no_specific_requirements"] = True
        return score_data

    fulfillment_count = 0
    total_requirements = len(functional_reqs)

    for req in functional_reqs:
        fulfilled = False

        if "New implementation" in req and files_created:
            fulfilled = True
            score_data["strengths"].append("New functionality implemented")

        if "Bug fix" in req and files_modified:
            fulfilled = True
            score_data["strengths"].append("Bug fixes applied")

        if "Testing component" in req and implementation_results.get("tests_implemented"):
            fulfilled = True
            score_data["strengths"].append("Testing implemented")

        if "API integration" in req and any("api" in f.lower() for f in files_created):
            fulfilled = True
            score_data["strengths"].append("API integration implemented")

        if fulfilled:
            fulfillment_count += 1
        else:
            score_data["issues"].append(f"Requirement not fully addressed: {req}")

    score_data["score"] = fulfillment_count / total_requirements if total_requirements > 0 else 0.0
    score_data["details"]["fulfilled_count"] = fulfillment_count
    score_data["details"]["total_requirements"] = total_requirements

    return score_data


def _evaluate_implementation_quality(implementation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the quality of implementation."""

    score_data = {
        "score": 0.0,
        "details": {},
        "issues": [],
        "strengths": []
    }

    quality_indicators = 0
    total_indicators = 5  # Total quality indicators to check

    # Check if files were created (indicates active implementation)
    if implementation_results.get("files_created"):
        quality_indicators += 1
        score_data["strengths"].append("New files created successfully")
    else:
        score_data["issues"].append("No new files created")

    # Check if tests were implemented
    if implementation_results.get("tests_implemented"):
        quality_indicators += 1
        score_data["strengths"].append("Tests implemented")
    else:
        score_data["issues"].append("No tests implemented")

    # Check error handling
    errors = implementation_results.get("errors_encountered", [])
    if len(errors) == 0:
        quality_indicators += 1
        score_data["strengths"].append("No implementation errors")
    else:
        score_data["issues"].append(f"Implementation errors encountered: {len(errors)}")

    # Check implementation status
    status = implementation_results.get("implementation_status", "")
    if status in ["completed_successfully", "completed_needs_review"]:
        quality_indicators += 1
        score_data["strengths"].append("Implementation completed")
    else:
        score_data["issues"].append(f"Implementation status: {status}")

    # Check if warnings are minimal
    warnings = implementation_results.get("warnings", [])
    if len(warnings) <= 2:
        quality_indicators += 1
        score_data["strengths"].append("Minimal warnings")
    else:
        score_data["issues"].append(f"Multiple warnings: {len(warnings)}")

    score_data["score"] = quality_indicators / total_indicators
    score_data["details"]["quality_indicators_met"] = quality_indicators
    score_data["details"]["total_indicators"] = total_indicators

    return score_data


def _evaluate_test_results(test_results: Dict[str, Any],
                          implementation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate test coverage and results."""

    score_data = {
        "score": 0.0,
        "details": {},
        "issues": [],
        "strengths": []
    }

    if not test_results:
        # No test results available
        if implementation_results.get("tests_implemented"):
            score_data["score"] = 0.5  # Tests implemented but no results
            score_data["issues"].append("Tests implemented but no results available")
        else:
            score_data["score"] = 0.0
            score_data["issues"].append("No tests implemented or executed")
        return score_data

    # Evaluate based on test results
    tests_passed = test_results.get("tests_passed", 0)
    tests_failed = test_results.get("tests_failed", 0)
    total_tests = tests_passed + tests_failed

    if total_tests == 0:
        score_data["score"] = 0.0
        score_data["issues"].append("No tests executed")
        return score_data

    # Calculate test success rate
    success_rate = tests_passed / total_tests
    score_data["score"] = success_rate

    if success_rate == 1.0:
        score_data["strengths"].append("All tests passed")
    elif success_rate >= 0.8:
        score_data["strengths"].append("Most tests passed")
    else:
        score_data["issues"].append(f"Test failure rate: {1 - success_rate:.2%}")

    score_data["details"]["tests_passed"] = tests_passed
    score_data["details"]["tests_failed"] = tests_failed
    score_data["details"]["success_rate"] = success_rate

    return score_data


def _evaluate_build_status(build_status: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate build status."""

    score_data = {
        "score": 0.0,
        "details": {},
        "issues": [],
        "strengths": []
    }

    if not build_status:
        score_data["score"] = 0.5  # Neutral score if no build info
        score_data["details"]["no_build_info"] = True
        return score_data

    if build_status.get("success", False):
        score_data["score"] = 1.0
        score_data["strengths"].append("Build successful")
    else:
        score_data["score"] = 0.0
        score_data["issues"].append("Build failed")
        if build_status.get("error"):
            score_data["issues"].append(f"Build error: {build_status['error']}")

    score_data["details"]["build_status"] = build_status

    return score_data


def _evaluate_error_rate(implementation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate error rate during implementation."""

    score_data = {
        "score": 0.0,
        "details": {},
        "issues": [],
        "strengths": []
    }

    errors = implementation_results.get("errors_encountered", [])
    warnings = implementation_results.get("warnings", [])

    total_issues = len(errors) + len(warnings)

    if total_issues == 0:
        score_data["score"] = 0.0  # No errors (good)
        score_data["strengths"].append("No errors or warnings")
    elif total_issues <= 2:
        score_data["score"] = 0.1  # Minimal errors
        score_data["strengths"].append("Minimal issues encountered")
    elif total_issues <= 5:
        score_data["score"] = 0.3  # Moderate errors
        score_data["issues"].append("Moderate number of issues")
    else:
        score_data["score"] = 0.5  # High error rate
        score_data["issues"].append("High number of issues")

    score_data["details"]["error_count"] = len(errors)
    score_data["details"]["warning_count"] = len(warnings)
    score_data["details"]["total_issues"] = total_issues

    return score_data


def _generate_recommendations(evaluation_results: Dict[str, Any],
                            requirements_analysis: Dict[str, Any],
                            implementation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommendations based on evaluation."""

    recommendations = {
        "immediate_actions": [],
        "improvements": [],
        "future_considerations": [],
        "priority": "medium"
    }

    overall_score = evaluation_results.get("overall_score", 0.0)

    # High priority issues
    if overall_score < 0.5:
        recommendations["priority"] = "high"
        recommendations["immediate_actions"].append("Address critical implementation issues")

    # Specific recommendations based on scores
    if evaluation_results.get("test_coverage", 0.0) < 0.7:
        recommendations["improvements"].append("Increase test coverage")

    if evaluation_results.get("build_success", 0.0) < 1.0:
        recommendations["immediate_actions"].append("Fix build issues")

    if evaluation_results.get("requirement_fulfillment", 0.0) < 0.8:
        recommendations["immediate_actions"].append("Complete unfulfilled requirements")

    if evaluation_results.get("error_rate", 0.0) > 0.3:
        recommendations["improvements"].append("Improve error handling and code quality")

    # Future considerations
    if overall_score >= 0.8:
        recommendations["future_considerations"].append("Consider performance optimization")
        recommendations["future_considerations"].append("Add additional documentation")

    # If no specific issues found
    if not recommendations["immediate_actions"] and not recommendations["improvements"]:
        recommendations["improvements"].append("Implementation meets requirements")
        recommendations["priority"] = "low"

    return recommendations


def _determine_final_status(evaluation_results: Dict[str, Any]) -> str:
    """Determine final status based on evaluation."""

    overall_score = evaluation_results.get("overall_score", 0.0)

    if overall_score >= 0.9:
        return "completed"
    elif overall_score >= 0.7:
        return "completed_with_minor_issues"
    elif overall_score >= 0.5:
        return "completed_needs_improvement"
    else:
        return "failed_requires_rework"


def _determine_next_actions(evaluation_results: Dict[str, Any],
                          recommendations: Dict[str, Any]) -> List[str]:
    """Determine next actions based on evaluation and recommendations."""

    next_actions = []

    # Add immediate actions from recommendations
    next_actions.extend(recommendations.get("immediate_actions", []))

    # Add based on overall score
    overall_score = evaluation_results.get("overall_score", 0.0)

    if overall_score >= 0.8:
        next_actions.append("Implementation ready for deployment")
    elif overall_score >= 0.6:
        next_actions.append("Address identified issues before deployment")
    else:
        next_actions.append("Significant rework required")

    # Default action if none specified
    if not next_actions:
        next_actions.append("Review implementation and provide feedback")

    return next_actions