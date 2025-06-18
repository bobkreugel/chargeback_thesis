from abc import ABC, abstractmethod
from typing import Dict, Any, List
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
    
    def _select_merchants(
        self, 
        num_transactions: int, 
        merchants: Dict[str, Any],
        reuse_probability: float
    ) -> List[str]:
        """
        Select merchants for transactions with given reuse probability.
        
        Args:
            num_transactions: Number of merchants to select
            merchants: Dictionary of available merchants
            reuse_probability: Probability of reusing the previous merchant
            
        Returns:
            List of merchant IDs
        """
        merchant_ids = list(merchants.keys())
        if not merchant_ids:
            raise ValueError("No merchants available")
            
        merchant_sequence = []
        
        # First transaction: randomly select a merchant
        current_merchant = random.choice(merchant_ids)
        merchant_sequence.append(current_merchant)
        
        # For each subsequent transaction
        for _ in range(num_transactions - 1):
            if random.random() < reuse_probability:
                # Reuse the same merchant as previous transaction
                merchant_sequence.append(current_merchant)
            else:
                # Select a new random merchant
                new_merchant = random.choice(merchant_ids)
                while new_merchant == current_merchant and len(merchant_ids) > 1:
                    new_merchant = random.choice(merchant_ids)
                # Update current merchant to the new one
                current_merchant = new_merchant
                merchant_sequence.append(current_merchant)
        
        return merchant_sequence
    
    def _select_single_merchant(
        self,
        merchants: Dict[str, Any],
        reuse_probability: float,
        previous_merchant: str = None,
        used_merchants: set = None
    ) -> str:
        """
        Select a single merchant with reuse consideration.
        
        Args:
            merchants: Dictionary of available merchants
            previous_merchant: Previously used merchant ID
            used_merchants: Set of already used merchants in this pattern
            reuse_probability: Probability of reusing the previous merchant
            
        Returns:
            Selected merchant ID
        """
        merchant_ids = list(merchants.keys())
        if not merchant_ids:
            raise ValueError("No merchants available")
            
        if used_merchants is None:
            used_merchants = set()
            
        # If we have a previous merchant and should reuse with given probability
        if previous_merchant and random.random() < reuse_probability:
            return previous_merchant
        
        # Select a new merchant
        return random.choice(merchant_ids)

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