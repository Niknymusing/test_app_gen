"""
MEU Framework Coder Agent Backend

This module contains the FastAPI backend implementation for the MEU coder agent.
"""

from .api import create_app

__all__ = ["create_app"]