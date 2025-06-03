from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Tuple, Set
import networkx as nx
from faker import Faker
from .base_pattern import BasePattern
import logging

logger = logging.getLogger(__name__)

class SerialChargebackPattern:
    """Pattern generator for serial chargebacks (multiple chargebacks from same customer)."""
    
    def __init__(self, config, seed: int = None):
        """Initialize the serial chargeback pattern generator."""
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
        Inject serial chargeback patterns into the transaction graph.
        
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
        logger.info(f"Generating {num_patterns} serial chargeback patterns")
        
        # Create copy of graph to modify
        graph = graph.copy()
        
        # Generate patterns
        for _ in range(num_patterns):
            # Create fraudulent customer with realistic data
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_fraudster_data()

            # Voeg expliciet 'is_fraudster': True toe aan de node-attributen
            graph.add_node(
                customer_id,
                node_type='customer',
                **{**customer_data, 'is_fraudster': True}
            )
            
            # Create card for customer
            card_id = str(uuid.uuid4())
            card_type = random.choice(['visa', 'mastercard', 'amex'])
            graph.add_node(
                card_id,
                node_type='card',
                card_type=card_type
            )
            graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
            
            # Select random merchant (high chance of reuse)
            merchant_id = random.choice(list(merchants.keys()))
            
            # Generate transactions within time window
            num_transactions = random.randint(
                self.config['transactions_in_pattern']['min'],
                self.config['transactions_in_pattern']['max']
            )
            
            # Calculate pattern timeframe
            pattern_days = random.randint(
                self.config['time_window']['min'],
                self.config['time_window']['max']
            )
            pattern_start = random.randint(0, int((end_date - start_date).total_seconds()))
            pattern_start = start_date + timedelta(seconds=pattern_start)
            pattern_end = pattern_start + timedelta(days=pattern_days)
            
            for _ in range(num_transactions):
                # Generate transaction timestamp
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
                    amount=round(random.uniform(50, 1000), 2),  # Use normal transaction amounts
                    timestamp=timestamp,
                    is_chargeback=False
                )
                
                # Add edges
                graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
                
                # Create chargeback
                delay = random.randint(
                    self.config['chargeback_delay']['min'],
                    self.config['chargeback_delay']['max']
                )
                
                chargeback_id = str(uuid.uuid4())
                graph.add_node(
                    chargeback_id,
                    node_type='transaction',
                    amount=graph.nodes[transaction_id]['amount'],
                    timestamp=timestamp + timedelta(days=delay),
                    is_chargeback=True,
                    original_transaction=transaction_id
                )
                
                # Add edges (same as original transaction)
                graph.add_edge(card_id, chargeback_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, chargeback_id, edge_type='merchant_to_transaction')
                
                # Decide whether to reuse merchant for next transaction
                if random.random() >= self.config['merchant_reuse_prob']:
                    merchant_id = random.choice(list(merchants.keys()))
        
        return graph 