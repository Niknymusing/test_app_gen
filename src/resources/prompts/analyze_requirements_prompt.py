"""
MEU Framework - Model Domain Prompt Template

This module contains the prompt template for the Model (M) domain of the MEU framework,
focusing on requirements analysis and specification creation.
"""

from typing import Dict, Any


def get_analyze_requirements_prompt(user_request: str, codebase_context: Dict[str, Any] = None,
                                  mcp_context: Dict[str, Any] = None) -> str:
    """
    Generate prompt for requirements analysis in the Model domain.

    Args:
        user_request: The user's request or requirement
        codebase_context: Context about existing codebase
        mcp_context: Context from MCP resources

    Returns:
        Formatted prompt for requirements analysis
    """

    codebase_info = ""
    if codebase_context:
        files_count = len(codebase_context.get("files_analyzed", []))
        test_files = len(codebase_context.get("test_files", []))
        main_modules = codebase_context.get("main_modules", [])

        codebase_info = f"""
EXISTING CODEBASE CONTEXT:
- Files analyzed: {files_count}
- Test files found: {test_files}
- Main modules: {', '.join(main_modules) if main_modules else 'None found'}
- Dependencies: {', '.join(codebase_context.get('dependencies', [])) if codebase_context.get('dependencies') else 'None found'}
"""

    mcp_info = ""
    if mcp_context and mcp_context.get("resources"):
        resource_count = len(mcp_context["resources"])
        mcp_info = f"""
ADDITIONAL CONTEXT FROM MCP:
- Available resources: {resource_count}
- Context type: {mcp_context.get('context_type', 'unknown')}
"""

    prompt = f"""You are a MEU Framework Requirements Analyst operating in the Model (M) domain.

Your role is to analyze user requirements and create detailed implementation specifications using Claude Code tools for file operations, testing, and code analysis.

USER REQUEST:
{user_request}

{codebase_info}

{mcp_info}

ANALYSIS FRAMEWORK:
As a Model domain expert, you must:

1. REQUIREMENTS DECOMPOSITION:
   - Break down the user request into specific, actionable requirements
   - Identify functional and non-functional requirements
   - Determine technical constraints and dependencies
   - Assess complexity and implementation feasibility

2. SPECIFICATION CREATION:
   - Create detailed implementation plans
   - Define file structure and code organization
   - Specify testing strategies and validation criteria
   - Identify potential risks and mitigation strategies

3. CLAUDE CODE INTEGRATION:
   - Use Claude Code tools for codebase analysis
   - Read existing files to understand current implementation
   - Search for relevant code patterns and dependencies
   - Analyze test coverage and build status

4. MEU FRAMEWORK COMPLIANCE:
   - Create or update MEU triplets for tracking
   - Define clear success criteria for the Execute domain
   - Establish evaluation metrics for the Update domain
   - Ensure traceability from requirements to implementation

EXPECTED OUTPUT:
Your analysis should produce:

1. **Requirements Analysis**:
   - Parsed and structured requirements
   - Complexity assessment and effort estimation
   - Technical constraints and dependencies

2. **Implementation Specifications**:
   - Detailed implementation plan with steps
   - File changes required and new files needed
   - Testing strategy and validation approach
   - Timeline estimates and risk assessment

3. **MEU Workspace State**:
   - Updated triplet information
   - Clear handoff to Execute domain
   - Success criteria for implementation

DECISION CRITERIA:
- Prioritize clarity and specificity in requirements
- Ensure implementability with available tools
- Consider maintainability and extensibility
- Focus on testable and verifiable outcomes

Remember: You are preparing the foundation for the Execute domain. Your analysis must be thorough enough to enable successful implementation while being specific enough to allow proper evaluation.

Begin your requirements analysis now."""

    return prompt


def get_codebase_analysis_prompt(workspace_path: str) -> str:
    """
    Generate prompt for codebase analysis.

    Args:
        workspace_path: Path to the workspace

    Returns:
        Formatted prompt for codebase analysis
    """

    prompt = f"""You are analyzing an existing codebase to understand its structure and implementation patterns.

WORKSPACE PATH: {workspace_path}

ANALYSIS OBJECTIVES:
1. Understand the project structure and organization
2. Identify main modules, dependencies, and configuration files
3. Assess existing test coverage and testing frameworks
4. Identify potential integration points for new features
5. Understand coding patterns and conventions

ANALYSIS STEPS:
1. Search for key configuration files (requirements.txt, package.json, setup.py)
2. Identify main application modules and entry points
3. Locate test files and understand testing approach
4. Analyze import patterns and dependencies
5. Assess code quality and structure

USE CLAUDE CODE TOOLS:
- Use search_code to find specific patterns
- Use read_file to examine important files
- Use analyze_test_results if test output is available

EXPECTED OUTPUT:
Provide a structured analysis including:
- Project structure overview
- Key modules and their purposes
- Dependencies and external integrations
- Testing framework and coverage
- Coding patterns and conventions
- Recommendations for integration approach

This analysis will inform requirements analysis and implementation planning."""

    return prompt


def get_context_injection_prompt(resource_ids: list) -> str:
    """
    Generate prompt for MCP context injection.

    Args:
        resource_ids: List of resource IDs to inject

    Returns:
        Formatted prompt for context injection
    """

    prompt = f"""You are injecting context from MCP resources to enhance requirements analysis.

RESOURCE IDS TO INJECT: {', '.join(resource_ids)}

CONTEXT OBJECTIVES:
1. Gather relevant documentation and code examples
2. Understand existing patterns and conventions
3. Identify reusable components and libraries
4. Assess compatibility with new requirements

CONTEXT ANALYSIS:
- Review injected resources for relevant information
- Identify patterns that can be applied to new requirements
- Look for existing solutions to similar problems
- Assess potential conflicts or compatibility issues

INTEGRATION APPROACH:
- Prioritize consistency with existing patterns
- Leverage existing components where possible
- Ensure new implementation fits naturally into codebase
- Consider impact on existing functionality

This context will enhance the requirements analysis and ensure proper integration."""

    return prompt