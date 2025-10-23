"""
MEU Framework - Update Domain Prompt Template

This module contains the prompt template for the Update (U) domain of the MEU framework,
focusing on results evaluation and feedback generation.
"""

from typing import Dict, Any


def get_evaluate_results_prompt(requirements_analysis: Dict[str, Any],
                               implementation_results: Dict[str, Any],
                               test_results: Dict[str, Any],
                               build_status: Dict[str, Any]) -> str:
    """
    Generate prompt for results evaluation in the Update domain.

    Args:
        requirements_analysis: Original requirements analysis
        implementation_results: Results from implementation
        test_results: Test execution results
        build_status: Build status information

    Returns:
        Formatted prompt for results evaluation
    """

    # Extract key metrics
    files_created = len(implementation_results.get("files_created", []))
    files_modified = len(implementation_results.get("files_modified", []))
    tests_implemented = len(implementation_results.get("tests_implemented", []))
    errors_encountered = len(implementation_results.get("errors_encountered", []))

    test_summary = ""
    if test_results:
        tests_passed = test_results.get("tests_passed", 0)
        tests_failed = test_results.get("tests_failed", 0)
        test_summary = f"Tests Passed: {tests_passed}, Tests Failed: {tests_failed}"
    else:
        test_summary = "No test results available"

    build_summary = "Build Successful" if build_status.get("success", False) else "Build Failed"

    prompt = f"""You are a MEU Framework Evaluation Specialist operating in the Update (U) domain.

Your role is to evaluate implementation results, provide comprehensive feedback, and determine next actions based on the complete MEU cycle execution.

ORIGINAL REQUIREMENTS:
Functional Requirements: {len(requirements_analysis.get("functional_requirements", []))} items
Complexity: {requirements_analysis.get("complexity_assessment", "unknown")}
Expected Effort: {requirements_analysis.get("estimated_effort", "unknown")}

IMPLEMENTATION RESULTS:
Files Created: {files_created}
Files Modified: {files_modified}
Tests Implemented: {tests_implemented}
Errors Encountered: {errors_encountered}
Implementation Status: {implementation_results.get("implementation_status", "unknown")}

TEST RESULTS:
{test_summary}

BUILD STATUS:
{build_summary}

EVALUATION FRAMEWORK:
As an Update domain specialist, you must:

1. COMPREHENSIVE EVALUATION:
   - Assess requirement fulfillment completeness
   - Evaluate implementation quality and standards
   - Analyze test coverage and effectiveness
   - Review build success and stability
   - Calculate overall success metrics

2. QUALITY ASSESSMENT:
   - Code quality and maintainability
   - Error handling and robustness
   - Performance and efficiency
   - Security and best practices
   - Documentation completeness

3. FEEDBACK GENERATION:
   - Identify strengths and achievements
   - Highlight areas for improvement
   - Provide specific recommendations
   - Suggest next actions and priorities

4. MEU FRAMEWORK COMPLIANCE:
   - Evaluate triplet completion status
   - Update workspace state with final results
   - Prepare feedback for stakeholders
   - Plan future iterations if needed

EVALUATION CRITERIA:

1. **Requirement Fulfillment** (0-1.0):
   - Were all functional requirements implemented?
   - Do implementations meet specified criteria?
   - Are non-functional requirements satisfied?

2. **Implementation Quality** (0-1.0):
   - Code follows project conventions
   - Proper error handling implemented
   - Good test coverage achieved
   - Documentation is adequate

3. **Test Coverage** (0-1.0):
   - Comprehensive test suite created
   - All tests pass successfully
   - Edge cases properly covered
   - Integration tests included

4. **Build Success** (0-1.0):
   - Build completes without errors
   - No breaking changes introduced
   - Dependencies properly managed
   - Deployment readiness achieved

5. **Error Rate** (0-1.0):
   - Minimal errors during implementation
   - Good error recovery and handling
   - Warnings addressed appropriately
   - Stable operation demonstrated

SCORING METHODOLOGY:
- Calculate individual scores for each criterion
- Compute weighted overall score
- Identify critical issues requiring immediate attention
- Highlight exceptional achievements

EVALUATION OUTPUTS:

1. **Quantitative Assessment**:
   - Overall success score (0-1.0)
   - Individual criterion scores
   - Comparative analysis with expectations
   - Trend analysis if applicable

2. **Qualitative Feedback**:
   - Detailed strengths and achievements
   - Specific areas for improvement
   - Root cause analysis for issues
   - Best practices observed

3. **Recommendations**:
   - Immediate actions required
   - Long-term improvements suggested
   - Process optimization opportunities
   - Future considerations

4. **Next Actions**:
   - Deployment readiness assessment
   - Additional testing requirements
   - Documentation updates needed
   - Stakeholder communication plan

DECISION FRAMEWORK:
- **Completed (0.9+)**: Ready for deployment
- **Minor Issues (0.7-0.89)**: Address issues before deployment
- **Needs Improvement (0.5-0.69)**: Significant work required
- **Requires Rework (<0.5)**: Major issues, consider restart

Remember: Your evaluation determines the success of the entire MEU cycle and guides future improvements. Be thorough, objective, and constructive in your assessment.

Begin comprehensive evaluation now."""

    return prompt


def get_quality_assessment_prompt(implementation_results: Dict[str, Any]) -> str:
    """
    Generate prompt for detailed quality assessment.

    Args:
        implementation_results: Results from implementation

    Returns:
        Formatted prompt for quality assessment
    """

    prompt = f"""You are performing detailed quality assessment of implementation results.

IMPLEMENTATION ARTIFACTS:
Files Created: {implementation_results.get("files_created", [])}
Files Modified: {implementation_results.get("files_modified", [])}
Tests Implemented: {implementation_results.get("tests_implemented", [])}

QUALITY DIMENSIONS TO ASSESS:

1. **Code Quality**:
   - Readability and maintainability
   - Adherence to coding standards
   - Proper naming conventions
   - Code organization and structure

2. **Functionality**:
   - Requirements implementation completeness
   - Feature correctness and reliability
   - Edge case handling
   - Performance characteristics

3. **Testing**:
   - Test coverage adequacy
   - Test quality and reliability
   - Test maintainability
   - Integration test completeness

4. **Documentation**:
   - Code documentation quality
   - API documentation completeness
   - Usage examples provided
   - Maintenance instructions

5. **Security**:
   - Security best practices followed
   - Input validation implemented
   - Error information disclosure
   - Authentication and authorization

ASSESSMENT METHODOLOGY:
- Use Claude Code tools to examine implementation
- Read created and modified files for quality review
- Analyze test implementations for completeness
- Check for security vulnerabilities and best practices

QUALITY SCORING:
- Rate each dimension on 0-1.0 scale
- Provide specific examples for scores
- Identify best practices demonstrated
- Highlight areas needing improvement

EXPECTED OUTPUT:
- Detailed quality scores by dimension
- Specific strengths and weaknesses
- Recommendations for improvement
- Overall quality assessment

Perform thorough quality assessment now."""

    return prompt


def get_recommendation_prompt(evaluation_results: Dict[str, Any]) -> str:
    """
    Generate prompt for creating recommendations.

    Args:
        evaluation_results: Results from evaluation

    Returns:
        Formatted prompt for recommendation generation
    """

    overall_score = evaluation_results.get("overall_score", 0.0)
    issues_found = evaluation_results.get("issues_found", [])
    strengths = evaluation_results.get("strengths", [])

    prompt = f"""You are generating recommendations based on comprehensive evaluation results.

EVALUATION SUMMARY:
Overall Score: {overall_score:.2f}
Issues Found: {len(issues_found)}
Strengths Identified: {len(strengths)}

MAJOR ISSUES:
{chr(10).join(f"- {issue}" for issue in issues_found[:5])}

KEY STRENGTHS:
{chr(10).join(f"- {strength}" for strength in strengths[:5])}

RECOMMENDATION FRAMEWORK:

1. **Immediate Actions** (High Priority):
   - Critical issues requiring immediate attention
   - Blocking problems preventing deployment
   - Security vulnerabilities to address
   - Build failures to resolve

2. **Improvements** (Medium Priority):
   - Code quality enhancements
   - Test coverage improvements
   - Documentation updates
   - Performance optimizations

3. **Future Considerations** (Low Priority):
   - Architectural improvements
   - Technology upgrades
   - Process optimizations
   - Feature enhancements

RECOMMENDATION CATEGORIES:

1. **Technical Recommendations**:
   - Code refactoring suggestions
   - Architecture improvements
   - Technology stack optimizations
   - Testing strategy enhancements

2. **Process Recommendations**:
   - Development workflow improvements
   - Quality assurance enhancements
   - Documentation standards
   - Deployment procedures

3. **Strategic Recommendations**:
   - Long-term technical debt management
   - Scalability considerations
   - Maintainability improvements
   - Team skill development

PRIORITIZATION CRITERIA:
- Impact on system reliability
- Implementation effort required
- Risk mitigation value
- User experience improvement

EXPECTED OUTPUT:
- Categorized and prioritized recommendations
- Clear action items with rationale
- Implementation effort estimates
- Success criteria for each recommendation

Generate comprehensive, actionable recommendations now."""

    return prompt


def get_final_status_prompt(evaluation_summary: Dict[str, Any]) -> str:
    """
    Generate prompt for determining final status.

    Args:
        evaluation_summary: Complete evaluation summary

    Returns:
        Formatted prompt for final status determination
    """

    prompt = f"""You are determining the final status of the MEU framework execution cycle.

EVALUATION SUMMARY:
Overall Score: {evaluation_summary.get("success_rate", 0.0):.2f}
Areas for Improvement: {len(evaluation_summary.get("areas_for_improvement", []))}
Recommendations Priority: {evaluation_summary.get("recommendations", {}).get("priority", "unknown")}

STATUS DETERMINATION CRITERIA:

1. **COMPLETED** (0.9+ score):
   - All requirements fully implemented
   - All tests passing
   - Build successful
   - Ready for deployment
   - Minimal issues identified

2. **COMPLETED WITH MINOR ISSUES** (0.7-0.89 score):
   - Requirements mostly implemented
   - Most tests passing
   - Build successful with minor warnings
   - Some improvements recommended
   - Non-blocking issues only

3. **COMPLETED NEEDS IMPROVEMENT** (0.5-0.69 score):
   - Core requirements implemented
   - Some test failures or gaps
   - Build issues present
   - Significant improvements needed
   - Some blocking issues

4. **FAILED REQUIRES REWORK** (<0.5 score):
   - Major requirements not met
   - Significant test failures
   - Build failures
   - Critical issues present
   - Substantial rework needed

ADDITIONAL CONSIDERATIONS:
- Security vulnerability presence
- Performance requirements met
- Scalability concerns addressed
- Maintainability standards achieved
- User experience quality

NEXT ACTIONS DETERMINATION:
Based on status, determine appropriate next actions:
- Deployment readiness
- Additional development needed
- Quality improvements required
- Complete rework necessary

STAKEHOLDER COMMUNICATION:
- Clear status explanation
- Specific issues and achievements
- Timeline for resolution
- Resource requirements

Determine final status and next actions now."""

    return prompt