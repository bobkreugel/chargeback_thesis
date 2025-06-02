from abc import ABC, abstractmethod
from typing import Dict, Any
import networkx as nx

class BasePattern(ABC):
    """Base class for all transaction patterns that can be injected into the dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern with configuration.
        
        Args:
            config: Pattern-specific configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def inject(self, graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        """
        Inject the pattern into the given transaction graph.
        
        Args:
            graph: The transaction graph to inject the pattern into
            **kwargs: Additional arguments needed for pattern injection
            
        Returns:
            The modified graph with the pattern injected
        """
        pass 