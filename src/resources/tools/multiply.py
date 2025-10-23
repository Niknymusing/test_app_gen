"""
Some simple example tools to showcase how we can enable tool calls on our agent
"""

from langchain_core.tools import tool


@tool("multiply", parse_docstring=True)
def multiply(x: float, y: float) -> str:
    """
    Tool that can be used to multiply two numbers and return the result as a string.

    Args:
        x: The first number.
        y: The second number.

    Returns:
        A string stating the multiplication result.
    """
    result = x * y
    return f"The result of {x} multiplied by {y} is {result}."
