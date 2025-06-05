from abc import ABC, abstractmethod
from typing import Dict, Any
import networkx as nx
from faker import Faker
import random
import ipaddress

class BasePattern(ABC):
    """Base class for all transaction patterns that can be injected into the dataset."""
    
    def __init__(self, config: Dict[str, Any], seed: int = None):
        """
        Initialize the pattern with configuration.
        
        Args:
            config: Pattern-specific configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
    
    def _generate_fraudster_data(self) -> Dict[str, Any]:
        """Generate data for a fraudulent customer."""
        return {
            'name': self.fake.name(),
            'email': self.fake.email(),
            'phone': self.fake.phone_number(),
            'address': self.fake.address().replace('\n', ', '),
            'city': self.fake.city(),
            'state': self.fake.state(),
            'zip': self.fake.zipcode(),
            'risk_score': round(random.uniform(60, 100), 2),  # Higher risk score for fraudsters
            'is_fraudster': True
        }
    
    def _generate_ip_address(self) -> str:
        """Generate a random IP address that looks realistic."""
        
        ranges = [
            (int(ipaddress.IPv4Address('5.0.0.0')), int(ipaddress.IPv4Address('95.255.255.255'))),
            (int(ipaddress.IPv4Address('104.0.0.0')), int(ipaddress.IPv4Address('191.255.255.255')))
        ]
        
        range_idx = random.randint(0, 1)
        start, end = ranges[range_idx]
        ip_int = random.randint(start, end)
        return str(ipaddress.IPv4Address(ip_int))
    
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