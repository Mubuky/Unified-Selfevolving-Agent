"""
Abstract Base Class for ExpeL Retrieval Systems

This module defines the interface for all retrieval systems used in the ExpeL framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import VectorStore

class BaseRetrieval(ABC):
    """
    Abstract base class for ExpeL retrieval systems.

    This class defines the interface for managing experience-based retrieval,
    including vector store setup, query construction, and similarity search.
    """

    def __init__(self,
                 embedder: Embeddings,
                 benchmark_name: str,
                 fewshot_strategy: str,
                 reranker: str,
                 buffer_retrieve_ratio: int,
                 max_fewshot_tokens: int,
                 num_fewshots: int,
                 message_splitter: Callable,
                 identifier: Callable,
                 step_stripper: Callable,
                 message_step_splitter: Callable,
                 remove_task_suffix: Callable,
                 token_counter: Callable):
        """
        Initialize the retrieval system.

        Args:
            embedder: Embedding model for vector encoding
            benchmark_name: Name of the benchmark (alfworld, webshop, etc.)
            fewshot_strategy: Strategy for selecting few-shot examples
            reranker: Re-ranking strategy for retrieved documents
            buffer_retrieve_ratio: Ratio for retrieving extra candidates
            max_fewshot_tokens: Maximum tokens allowed per few-shot example
            num_fewshots: Number of few-shot examples to retrieve
            message_splitter: Function to split trajectory messages
            identifier: Function to identify message types
            step_stripper: Function to extract clean content from steps
            message_step_splitter: Function to split trajectory into steps
            remove_task_suffix: Function to clean task descriptions
            token_counter: Function to count tokens in text
        """
        self.embedder = embedder
        self.benchmark_name = benchmark_name
        self.fewshot_strategy = fewshot_strategy
        self.reranker = reranker
        self.buffer_retrieve_ratio = buffer_retrieve_ratio
        self.max_fewshot_tokens = max_fewshot_tokens
        self.num_fewshots = num_fewshots

        # Utility functions
        self.message_splitter = message_splitter
        self.identifier = identifier
        self.step_stripper = step_stripper
        self.message_step_splitter = message_step_splitter
        self.remove_task_suffix = remove_task_suffix
        self.token_counter = token_counter

        # Internal state
        self.docs: List[Document] = []
        self.combined_history: Dict[str, List[Any]] = {}
        self.keys2task: Dict[str, Dict] = {}
        self.vectorstore: Optional[VectorStore] = None

    @abstractmethod
    def setup_documents(self,
                       succeeded_trial_history: Dict[str, List[Any]],
                       all_fewshots: Any,
                       env: Any) -> None:
        """
        Setup documents from succeeded trial history and few-shot examples.

        Args:
            succeeded_trial_history: Dictionary of successful trajectories
            all_fewshots: Few-shot examples from the dataset
            env: Environment object for metadata extraction
        """
        pass

    @abstractmethod
    def build_query_vectors(self,
                           task: str,
                           trajectory: Any,
                           prompt_history: List) -> Dict[str, str]:
        """
        Build query vectors for retrieval.

        Args:
            task: Current task description
            trajectory: Current trajectory object
            prompt_history: Current prompt history

        Returns:
            Dictionary mapping query types to query strings
        """
        pass

    @abstractmethod
    def create_filtered_vectorstore(self,
                                   strategy: str,
                                   env_name: str) -> VectorStore:
        """
        Create a filtered vector store based on strategy and environment.

        Args:
            strategy: Filtering strategy (task_similarity, thought_similarity, etc.)
            env_name: Environment name for filtering

        Returns:
            Filtered vector store
        """
        pass

    @abstractmethod
    def retrieve_topk_documents(self,
                               queries: Dict[str, str],
                               query_type: str) -> List[str]:
        """
        Retrieve top-k most similar documents.

        Args:
            queries: Dictionary of query strings
            query_type: Type of query to use for retrieval

        Returns:
            List of retrieved few-shot examples
        """
        pass


    def get_retrieval_info(self) -> Dict[str, Any]:
        """
        Get information about the retrieval system.

        Returns:
            Dictionary with retrieval system information
        """
        return {
            'retrieval_type': self.__class__.__name__,
            'benchmark': self.benchmark_name,
            'fewshot_strategy': self.fewshot_strategy,
            'reranker': self.reranker,
            'num_documents': len(self.docs),
            'num_fewshots': self.num_fewshots,
            'max_fewshot_tokens': self.max_fewshot_tokens,
            'buffer_retrieve_ratio': self.buffer_retrieve_ratio
        }

    def __repr__(self) -> str:
        """String representation of the retrieval system."""
        return f"{self.__class__.__name__}(benchmark={self.benchmark_name}, strategy={self.fewshot_strategy})"