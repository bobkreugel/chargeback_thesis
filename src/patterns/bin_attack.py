from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Set
import networkx as nx
import logging
from .base_pattern import BasePattern

logger = logging.getLogger(__name__)

class BINAttackPattern(BasePattern):
    """Pattern generator for BIN attacks (testing multiple cards with same BIN prefix)."""
    
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
                fraud_type='bin_attack',
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
            
            # Generate fraudulent cards with same BIN
            bin_prefix = random.choice(self.config['bin_prefixes'])
            
            # Calculate number of cards based on config
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
            
            # Select merchants based on reuse probability
            merchant_sequence = self._select_merchants(num_cards, merchants)
            
            # Generate transactions within time window
            pattern_start = random.randint(0, int((end_date - start_date).total_seconds()))
            pattern_start = start_date + timedelta(seconds=pattern_start)
            pattern_end = pattern_start + timedelta(minutes=self.config['time_window']['minutes'])
            
            for i, card_id in enumerate(card_ids):
                # Use merchant from sequence
                merchant_id = merchant_sequence[i]
                
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
        
        return graph 

    def _select_merchants(
        self,
        num_transactions: int,
        merchants: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Select merchants for transactions with given reuse rate.
        For each transaction after the first:
        - With probability merchant_reuse_prob: use the same merchant as previous transaction
        - With probability (1 - merchant_reuse_prob): select a new random merchant
        """
        merchant_ids = list(merchants.keys())
        merchant_sequence = []
        
        # First transaction: randomly select a merchant
        current_merchant = random.choice(merchant_ids)
        merchant_sequence.append(current_merchant)
        
        # For each subsequent transaction
        for _ in range(num_transactions - 1):
            if random.random() < self.config['merchant_reuse_prob']:
                # Reuse the same merchant as previous transaction
                merchant_sequence.append(current_merchant)
            else:
                # Select a new random merchant
                new_merchant = random.choice(merchant_ids)
                while new_merchant == current_merchant and len(merchant_ids) > 1:
                    new_merchant = random.choice(merchant_ids)
                current_merchant = new_merchant
                merchant_sequence.append(current_merchant)
        
        return merchant_sequence 