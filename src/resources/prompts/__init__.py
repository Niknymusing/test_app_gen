"""
MEU Framework Prompt Templates

This module contains prompt templates for the MEU (Model-Execute-Update)
framework workflow in the coder agent.
"""

from .analyze_requirements_prompt import (
    get_analyze_requirements_prompt,
    get_codebase_analysis_prompt,
    get_context_injection_prompt
)
from .implement_solution_prompt import (
    get_implement_solution_prompt,
    get_step_implementation_prompt,
    get_testing_implementation_prompt,
    get_error_recovery_prompt
)
from .evaluate_results_prompt import (
    get_evaluate_results_prompt,
    get_quality_assessment_prompt,
    get_recommendation_prompt,
    get_final_status_prompt
)

__all__ = [
    # Model domain prompts
    "get_analyze_requirements_prompt",
    "get_codebase_analysis_prompt",
    "get_context_injection_prompt",

    # Execute domain prompts
    "get_implement_solution_prompt",
    "get_step_implementation_prompt",
    "get_testing_implementation_prompt",
    "get_error_recovery_prompt",

    # Update domain prompts
    "get_evaluate_results_prompt",
    "get_quality_assessment_prompt",
    "get_recommendation_prompt",
    "get_final_status_prompt"
]