from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Tuple, Set
import networkx as nx
from faker import Faker
from .base_pattern import BasePattern
import logging
import ipaddress

logger = logging.getLogger(__name__)

class BINAttackPattern:
    """Pattern generator for BIN attacks (testing multiple cards with same BIN prefix)."""
    
    def __init__(self, config, seed: int = None):
        """Initialize the BIN attack pattern generator."""
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
        # Generate a random IP in these ranges:
        # 5.0.0.0 to 95.255.255.255 (European/Asian ISPs)
        # 104.0.0.0 to 191.255.255.255 (North American ISPs)
        # Avoid private ranges and other special purpose IPs
        
        ranges = [
            (int(ipaddress.IPv4Address('5.0.0.0')), int(ipaddress.IPv4Address('95.255.255.255'))),
            (int(ipaddress.IPv4Address('104.0.0.0')), int(ipaddress.IPv4Address('191.255.255.255')))
        ]
        
        range_idx = random.randint(0, 1)
        start, end = ranges[range_idx]
        ip_int = random.randint(start, end)
        return str(ipaddress.IPv4Address(ip_int))
    
    def inject(
        self,
        graph: nx.DiGraph,
        num_patterns: int,
        start_date: datetime,
        end_date: datetime,
        customers: Dict[str, Dict[str, Any]],
        customer_cards: Dict[str, Set[str]],
        merchants: Dict[str, Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        Inject BIN attack patterns into the transaction graph.
        
        Args:
            graph: The transaction graph to inject patterns into
            num_patterns: Number of patterns to inject
            start_date: Start of possible transaction dates
            end_date: End of possible transaction dates
            customers: Dictionary of customer IDs to their data
            customer_cards: Mapping of customer IDs to their card IDs
            merchants: Dictionary of merchant IDs to their data
            
        Returns:
            The modified graph with injected patterns
        """
        logger.info(f"Generating {num_patterns} BIN attack patterns")
        
        # Create copy of graph to modify
        graph = graph.copy()
        
        # Generate patterns
        for _ in range(num_patterns):
            # Create fraudulent customer with realistic data
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_fraudster_data()
            
            # Add customer node with all attributes
            graph.add_node(
                customer_id,
                node_type='customer',
                **customer_data
            )
            
            # Generate IP address for this attack pattern
            ip_address = self._generate_ip_address()
            
            # Add IP node
            ip_node_id = f"ip_{customer_id}"  # Use customer_id to make it unique
            graph.add_node(
                ip_node_id,
                node_type='ip',
                address=ip_address,
                is_fraudulent=True
            )
            
            # Link IP to customer
            graph.add_edge(ip_node_id, customer_id, edge_type='ip_to_customer')
            
            # Select random merchant (high chance of reuse)
            merchant_id = random.choice(list(merchants.keys()))
            
            # Generate fraudulent cards with same BIN
            bin_prefix = random.choice(self.config['bin_prefixes'])
            num_cards = random.randint(
                self.config['num_cards']['min'],
                self.config['num_cards']['max']
            )
            
            # Create fraudulent cards
            card_ids = []
            for _ in range(num_cards):
                card_id = f"{bin_prefix}{str(random.randint(0, 9999999999)).zfill(10)}"
                card_type = random.choice(['visa', 'mastercard', 'amex'])
                graph.add_node(
                    card_id,
                    node_type='card',
                    card_type=card_type,
                    is_fraudulent=True
                )
                graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
                card_ids.append(card_id)
            
            # Generate transactions within time window
            pattern_start = random.randint(0, int((end_date - start_date).total_seconds()))
            pattern_start = start_date + timedelta(seconds=pattern_start)
            pattern_end = pattern_start + timedelta(minutes=self.config['time_window']['minutes'])
            
            for card_id in card_ids:
                # Generate transaction
                amount = round(
                    random.uniform(
                        self.config['transaction_amount']['min'],
                        self.config['transaction_amount']['max']
                    ),
                    2
                )
                
                timestamp = pattern_start + timedelta(
                    seconds=random.randint(
                        0,
                        int((pattern_end - pattern_start).total_seconds())
                    )
                )
                
                # Create transaction node
                transaction_id = str(uuid.uuid4())
                graph.add_node(
                    transaction_id,
                    node_type='transaction',
                    amount=amount,
                    timestamp=timestamp,
                    is_chargeback=False,
                    source_ip=ip_address,  # Store the IP that made this transaction
                    customer_id=customer_id  # Store the customer ID directly in the transaction
                )
                
                # Add edges
                graph.add_edge(ip_node_id, transaction_id, edge_type='ip_to_transaction')
                graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
                
                # Add chargeback with configured probability
                if random.random() < self.config['chargeback_rate']:
                    delay = random.randint(
                        self.config['chargeback_delay']['min'],
                        self.config['chargeback_delay']['max']
                    )
                    
                    chargeback_id = str(uuid.uuid4())
                    graph.add_node(
                        chargeback_id,
                        node_type='transaction',
                        amount=amount,
                        timestamp=timestamp + timedelta(days=delay),
                        is_chargeback=True,
                        original_transaction=transaction_id,
                        source_ip=ip_address,  # Store the IP that made this transaction
                        customer_id=customer_id  # Store the customer ID directly in the transaction
                    )
                    
                    # Add edges (same as original transaction)
                    graph.add_edge(ip_node_id, chargeback_id, edge_type='ip_to_transaction')
                    graph.add_edge(card_id, chargeback_id, edge_type='card_to_transaction')
                    graph.add_edge(merchant_id, chargeback_id, edge_type='merchant_to_transaction')
                
                # Decide whether to reuse merchant for next transaction
                if random.random() >= self.config['merchant_reuse_prob']:
                    merchant_id = random.choice(list(merchants.keys()))
        
        return graph 