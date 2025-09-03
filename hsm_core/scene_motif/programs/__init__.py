"""
Programs Module

Domain-specific program interpreters and validators for scene motif generation.
"""

from .program import Program
from .interpretor import execute, execute_with_context
from .validator import validate_syntax

__all__ = [
    'Program',
    'execute',
    'execute_with_context',
    'validate_syntax',
]

