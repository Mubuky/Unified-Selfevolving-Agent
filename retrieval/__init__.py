"""
Retrieval System for ExpeL Framework

This module contains the retrieval system classes for managing
experience-based retrieval using FAISS and other vector stores.
"""

from .base import BaseRetrieval
from .expel_retrieval import ExpelRetrieval

__all__ = ['BaseRetrieval', 'ExpelRetrieval']