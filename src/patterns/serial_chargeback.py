from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Tuple
import networkx as nx
from faker import Faker
from .base_pattern import BasePattern
import logging

logger = logging.getLogger(__name__)

class SerialChargebackPattern(BasePattern):
    """
    Pattern that injects serial chargeback behavior into the transaction graph.
    This pattern creates clusters of chargebacks from the same customer within a short time window,
    potentially targeting the same merchant multiple times.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the serial chargeback pattern.
        
        Args:
            config: Configuration dictionary containing pattern settings
        """
        super().__init__(config)
        self.faker = Faker()
        
    def inject(
        self,
        graph: nx.MultiDiGraph,
        start_date: datetime,
        end_date: datetime,
        amount_range: Dict[str, float]
    ) -> nx.MultiDiGraph:
        """
        Inject serial chargeback patterns into the transaction graph.
        
        Args:
            graph: The transaction graph to inject patterns into
            start_date: Start of possible transaction dates
            end_date: End of possible transaction dates
            amount_range: Range for transaction amounts
            
        Returns:
            Modified graph with injected patterns
        """
        # Get all customer nodes
        customers = [
            node for node, attr in graph.nodes(data=True)
            if attr.get('node_type') == 'customer'
        ]
        
        # Calculate number of customers to inject pattern for
        num_pattern_customers = int(len(customers) * self.config['customer_ratio'])
        pattern_customers = random.sample(customers, num_pattern_customers)
        
        # Inject pattern for each selected customer
        for customer_id in pattern_customers:
            self._inject_pattern_for_customer(
                graph,
                customer_id,
                start_date,
                end_date,
                amount_range
            )
        
        return graph
    
    def _inject_pattern_for_customer(
        self,
        graph: nx.MultiDiGraph,
        customer_id: str,
        start_date: datetime,
        end_date: datetime,
        amount_range: Dict[str, float]
    ) -> None:
        """
        Inject a serial chargeback pattern for a specific customer.
        
        Args:
            graph: The transaction graph
            customer_id: ID of the customer to inject pattern for
            start_date: Start of possible transaction dates
            end_date: End of possible transaction dates
            amount_range: Range for transaction amounts
        """
        # Get customer's cards
        customer_cards = []
        for edge in graph.out_edges(customer_id, data=True):
            if edge[2].get('relationship_type') == 'HAS_CARD':
                customer_cards.append(edge[1])
        
        if not customer_cards:
            logger.warning(f"Customer {customer_id} has no cards, skipping pattern injection")
            return
        
        # Get all merchant nodes
        merchants = [
            node for node, attr in graph.nodes(data=True)
            if attr.get('node_type') == 'merchant'
        ]
        
        # Determine pattern parameters
        num_chargebacks = random.randint(
            self.config['chargebacks_in_pattern']['min'],
            self.config['chargebacks_in_pattern']['max']
        )
        
        # Select time window for pattern
        window_days = random.randint(
            self.config['time_window']['min'],
            self.config['time_window']['max']
        )
        
        # Select pattern start date
        pattern_start = self.faker.date_time_between(
            start_date=start_date,
            end_date=end_date - timedelta(days=window_days)
        )
        pattern_end = pattern_start + timedelta(days=window_days)
        
        # Initialize list to track used merchants
        used_merchants = []
        merchants_for_pattern = []
        
        # Select merchants for each transaction with proper reuse probability
        for _ in range(num_chargebacks):
            # If we have used merchants and random check passes, reuse a merchant
            if used_merchants and random.random() < self.config['repeat_merchant_prob']:
                merchant_id = random.choice(used_merchants)
            else:
                # Select a new merchant
                available_merchants = [m for m in merchants if m not in used_merchants]
                if not available_merchants:  # If we've used all merchants, reset the pool
                    available_merchants = merchants
                merchant_id = random.choice(available_merchants)
                used_merchants.append(merchant_id)
            
            merchants_for_pattern.append(merchant_id)
        
        # Generate transactions in pattern
        for i in range(num_chargebacks):
            # Select random card for this transaction
            card_id = random.choice(customer_cards)
            merchant_id = merchants_for_pattern[i]
            
            # Create transaction
            transaction_id = str(uuid.uuid4())
            transaction_date = self.faker.date_time_between(
                start_date=pattern_start,
                end_date=pattern_end
            )
            
            # Calculate chargeback date
            delay_days = random.randint(
                self.config['chargeback_delay']['min'],
                self.config['chargeback_delay']['max']
            )
            chargeback_date = transaction_date + timedelta(days=delay_days)
            
            # Select chargeback reason
            reason = random.choices(
                list(self.config['reasons'].keys()),
                weights=list(self.config['reasons'].values())
            )[0]
            
            # Create transaction node
            transaction_data = {
                'id': transaction_id,
                'amount': round(random.uniform(amount_range['min'], amount_range['max']), 2),
                'currency': 'USD',
                'timestamp': transaction_date.isoformat(),
                'status': 'chargeback',
                'is_chargeback': True,
                'chargeback_reason': reason,
                'chargeback_date': chargeback_date.isoformat(),
                'node_type': 'transaction'
            }
            
            # Add nodes and edges to graph
            graph.add_node(transaction_id, **transaction_data)
            graph.add_edge(
                card_id,
                transaction_id,
                relationship_type='MADE_TRANSACTION',
                timestamp=transaction_date.isoformat()
            )
            graph.add_edge(
                transaction_id,
                merchant_id,
                relationship_type='PAID_TO',
                timestamp=transaction_date.isoformat()
            ) 