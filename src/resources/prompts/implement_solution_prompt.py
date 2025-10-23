"""
MEU Framework - Execute Domain Prompt Template

This module contains the prompt template for the Execute (E) domain of the MEU framework,
focusing on solution implementation and code generation.
"""

from typing import Dict, Any


def get_implement_solution_prompt(requirements_analysis: Dict[str, Any],
                                implementation_specs: Dict[str, Any],
                                workspace_path: str) -> str:
    """
    Generate prompt for solution implementation in the Execute domain.

    Args:
        requirements_analysis: Analyzed requirements from Model domain
        implementation_specs: Implementation specifications
        workspace_path: Path to the workspace

    Returns:
        Formatted prompt for solution implementation
    """

    # Extract key information
    functional_reqs = requirements_analysis.get("functional_requirements", [])
    complexity = requirements_analysis.get("complexity_assessment", "medium")
    implementation_plan = implementation_specs.get("implementation_plan", [])
    test_strategy = implementation_specs.get("test_strategy", [])

    prompt = f"""You are a MEU Framework Implementation Specialist operating in the Execute (E) domain.

Your role is to implement solutions based on detailed specifications from the Model domain, using Claude Code tools for all file operations, testing, and build processes.

WORKSPACE PATH: {workspace_path}

REQUIREMENTS ANALYSIS:
Functional Requirements:
{chr(10).join(f"- {req}" for req in functional_reqs)}

Complexity Assessment: {complexity}
Estimated Effort: {requirements_analysis.get("estimated_effort", "moderate")}

IMPLEMENTATION SPECIFICATIONS:
Implementation Plan:
{chr(10).join(f"- {step}" for step in implementation_plan)}

Testing Strategy:
{chr(10).join(f"- {strategy}" for strategy in test_strategy)}

File Changes Required: {len(implementation_specs.get("file_changes_required", []))} files
New Files Needed: {len(implementation_specs.get("new_files_needed", []))} files

IMPLEMENTATION FRAMEWORK:
As an Execute domain specialist, you must:

1. CODE IMPLEMENTATION:
   - Follow the implementation plan step by step
   - Use Claude Code tools for all file operations
   - Implement new functionality as specified
   - Modify existing files only when necessary
   - Ensure code follows project conventions

2. TESTING IMPLEMENTATION:
   - Create comprehensive test suites
   - Implement unit tests for new functionality
   - Add integration tests where appropriate
   - Ensure tests are runnable and maintainable

3. QUALITY ASSURANCE:
   - Run tests after implementation
   - Check build status and fix issues
   - Handle errors gracefully
   - Provide clear error messages and logging

4. MEU FRAMEWORK COMPLIANCE:
   - Register tests in MEU framework
   - Update workspace state with progress
   - Track implementation artifacts
   - Prepare for Update domain evaluation

CLAUDE CODE TOOLS USAGE:
- write_file: Create new files with complete implementation
- edit_file: Modify existing files with targeted changes
- read_file: Read files to understand context before changes
- search_code: Find relevant code patterns and dependencies
- run_tests: Execute test suites after implementation
- check_build_status: Verify build integrity

IMPLEMENTATION GUIDELINES:
1. **Code Quality**:
   - Write clean, readable, and maintainable code
   - Follow existing coding patterns and conventions
   - Include appropriate comments and documentation
   - Handle edge cases and error conditions

2. **Testing Approach**:
   - Test-driven development where appropriate
   - Comprehensive test coverage for new functionality
   - Integration tests for complex interactions
   - Performance tests for critical paths

3. **Error Handling**:
   - Graceful error handling and recovery
   - Clear error messages for debugging
   - Logging for operational monitoring
   - Validation of inputs and outputs

4. **Documentation**:
   - Inline code documentation
   - Function and class docstrings
   - Usage examples where helpful
   - Update existing documentation as needed

VALIDATION CRITERIA:
- All tests pass successfully
- Build completes without errors
- Code follows project conventions
- Requirements are fully implemented
- No new security vulnerabilities introduced

PROGRESS TRACKING:
Track and report:
- Files created and modified
- Tests implemented and results
- Build status and any issues
- Implementation challenges and solutions

Remember: You are implementing the specifications created in the Model domain. Your implementation must be thorough, tested, and ready for evaluation in the Update domain.

Begin implementation now."""

    return prompt


def get_step_implementation_prompt(step: str, context: Dict[str, Any]) -> str:
    """
    Generate prompt for implementing a specific step.

    Args:
        step: The implementation step to execute
        context: Additional context for the step

    Returns:
        Formatted prompt for step implementation
    """

    prompt = f"""You are implementing a specific step in the Execute domain workflow.

IMPLEMENTATION STEP: {step}

CONTEXT:
{chr(10).join(f"- {k}: {v}" for k, v in context.items() if k != "step")}

STEP-SPECIFIC GUIDANCE:

1. If implementing NEW FUNCTIONALITY:
   - Design clean, modular code structure
   - Create comprehensive documentation
   - Implement thorough error handling
   - Write corresponding test cases

2. If fixing BUGS:
   - Identify root cause through code analysis
   - Apply targeted fixes without side effects
   - Add regression tests to prevent recurrence
   - Update documentation if needed

3. If implementing TESTING:
   - Create comprehensive test suites
   - Cover happy path and edge cases
   - Include integration and unit tests
   - Ensure tests are maintainable

4. If integrating APIs:
   - Implement robust error handling
   - Add proper authentication and security
   - Include retry logic and timeouts
   - Create comprehensive integration tests

EXECUTION REQUIREMENTS:
- Use Claude Code tools for all file operations
- Follow existing code patterns and conventions
- Ensure changes are backward compatible
- Test implementation thoroughly

EXPECTED OUTPUT:
- Complete implementation of the step
- Any files created or modified
- Test results and validation
- Documentation of changes made

Implement this step now with careful attention to quality and testing."""

    return prompt


def get_testing_implementation_prompt(test_strategy: list, workspace_path: str) -> str:
    """
    Generate prompt for implementing tests.

    Args:
        test_strategy: List of testing strategies to implement
        workspace_path: Path to the workspace

    Returns:
        Formatted prompt for test implementation
    """

    prompt = f"""You are implementing comprehensive testing in the Execute domain.

WORKSPACE PATH: {workspace_path}

TESTING STRATEGIES TO IMPLEMENT:
{chr(10).join(f"- {strategy}" for strategy in test_strategy)}

TESTING FRAMEWORK:
1. **Unit Tests**:
   - Test individual functions and methods
   - Mock external dependencies
   - Cover both success and failure cases
   - Test edge cases and boundary conditions

2. **Integration Tests**:
   - Test component interactions
   - Test external API integrations
   - Test database operations
   - Test file system operations

3. **End-to-End Tests**:
   - Test complete user workflows
   - Test system behavior under load
   - Test error recovery scenarios
   - Test configuration changes

TESTING BEST PRACTICES:
- Clear, descriptive test names
- Arrange-Act-Assert pattern
- Independent and isolated tests
- Fast execution and reliable results
- Easy to understand and maintain

TEST STRUCTURE:
- Use appropriate testing framework (pytest, unittest, etc.)
- Organize tests in logical modules
- Use fixtures for common test setup
- Include test utilities and helpers

VALIDATION:
- All tests should pass
- Tests should be runnable independently
- Good test coverage of new functionality
- Tests should be maintainable and readable

Use Claude Code tools to:
- Create test files with comprehensive coverage
- Run tests and analyze results
- Fix any test failures
- Verify test quality and coverage

Implement comprehensive testing now."""

    return prompt


def get_error_recovery_prompt(error_info: Dict[str, Any]) -> str:
    """
    Generate prompt for error recovery during implementation.

    Args:
        error_info: Information about the error encountered

    Returns:
        Formatted prompt for error recovery
    """

    prompt = f"""You encountered an error during implementation in the Execute domain.

ERROR INFORMATION:
Error Type: {error_info.get('type', 'Unknown')}
Error Message: {error_info.get('message', 'No message provided')}
Context: {error_info.get('context', 'No context provided')}
Step: {error_info.get('step', 'Unknown step')}

ERROR RECOVERY APPROACH:
1. **Analyze the Error**:
   - Understand the root cause
   - Identify affected components
   - Assess impact on implementation

2. **Recovery Strategy**:
   - Implement targeted fix
   - Ensure no side effects
   - Test the fix thoroughly
   - Document the solution

3. **Prevention**:
   - Add error handling for similar cases
   - Update tests to catch this error type
   - Improve input validation
   - Add logging for better debugging

RECOVERY STEPS:
1. Use Claude Code tools to analyze the problem
2. Read relevant files to understand context
3. Implement a targeted fix
4. Test the fix thoroughly
5. Update implementation to prevent recurrence

VALIDATION:
- Error is resolved completely
- No new errors introduced
- Tests pass after fix
- Implementation continues successfully

Resolve this error and continue with implementation."""

    return prompt