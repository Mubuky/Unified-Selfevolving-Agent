"""
ExpeL Storage Module

This module provides storage classes for managing the data persistence
and transfer chain in the ExpeL framework.
"""

from .base import BaseStorage
from .expel_storage import ExpelStorage

__all__ = ['BaseStorage', 'ExpelStorage']