from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Set
import networkx as nx
import logging
from .base_pattern import BasePattern

logger = logging.getLogger(__name__)

class SerialChargebackPattern(BasePattern):
    """Pattern generator for serial chargeback fraud."""
    
    def __init__(self, config, seed: int = None):
        """Initialize the serial chargeback pattern generator."""
        super().__init__(config, seed)
        self._global_merchant_usage = {}  # Track global merchant usage
    
    def _distribute_timestamps(
        self,
        num_transactions: int,
        pattern_start: datetime,
        pattern_days: int
    ) -> List[datetime]:
        """
        Distribute transactions across the time window.
        Each fraudster gets their own random window size between min and max days.
        All transactions will occur within exactly window_size days.
        """
        # Get time window from config
        window_min = self.config['time_window']['min']
        window_max = self.config['time_window']['max']
        
        # Choose a random window size for this fraudster
        window_size = random.randint(window_min, window_max)
        
        # First transaction at start
        timestamps = [pattern_start]
        
        # Last transaction at window_size days
        last_transaction = pattern_start + timedelta(days=window_size)
        timestamps.append(last_transaction)
        
        # Generate remaining transactions
        remaining_transactions = num_transactions - 2
        for _ in range(remaining_transactions):
            # Random day between 0 and window_size
            day = random.randint(0, window_size)
            # Random time within the day
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = pattern_start + timedelta(
                days=day,
                hours=hour,
                minutes=minute,
                seconds=second
            )
            timestamps.append(timestamp)
        
        # Sort timestamps to maintain chronological order
        timestamps.sort()
        
        return timestamps
    

    
    def inject(
        self,
        graph: nx.DiGraph,
        num_patterns: int,
        start_date: datetime,
        end_date: datetime,
        customers: Dict[str, Dict[str, Any]],
        customer_cards: Dict[str, Set[str]],
        merchants: Dict[str, Dict[str, Any]],
        global_config: Dict[str, Any] = None
    ) -> nx.DiGraph:
        """
        Inject serial chargeback patterns into the transaction graph.
        Each pattern consists of:
        1. Fraudulent transactions within the time window that get charged back
        2. Normal transactions within the same time window (no chargebacks)
        """
        logger.info(f"Generating {num_patterns} serial chargeback patterns")
        
        # Get time window and chargeback delay from config
        window_min = self.config['time_window']['min']
        window_max = self.config['time_window']['max']
        cb_delay_min = self.config['chargeback_delay']['min']
        cb_delay_max = self.config['chargeback_delay']['max']
        
        # Get global transaction amount range from global config if provided
        if global_config and 'transactions' in global_config:
            global_amount_min = global_config['transactions']['amount']['min']
            global_amount_max = global_config['transactions']['amount']['max']
        else:
            # Fallback to reasonable defaults if global config not available
            global_amount_min = 10.00
            global_amount_max = 500.00
        
        # Generate patterns
        for _ in range(num_patterns):
            # Create fraudulent customer with realistic data
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_fraudster_data()
            
            # Add customer node with all attributes
            graph.add_node(
                customer_id,
                node_type='customer',
                fraud_type='serial_chargeback',
                **customer_data
            )
            
            # Generate card for this customer
            card_id = str(uuid.uuid4())
            card_type = random.choice(['visa', 'mastercard', 'amex'])
            graph.add_node(
                card_id,
                node_type='card',
                card_type=card_type,
                is_fraudulent=True
            )
            graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
            
            # Calculate number of transactions in pattern (all will be fraudulent and get chargebacks)
            num_transactions = random.randint(
                self.config['transactions_in_pattern']['min'],
                self.config['transactions_in_pattern']['max']
            )
            
            # Calculate pattern start time, ensuring enough room for chargebacks
            max_delay = cb_delay_max + window_max  # Maximum days needed for pattern + chargebacks
            latest_start = int((end_date - start_date).total_seconds()) - (max_delay * 24 * 60 * 60)
            if latest_start < 0:
                logger.warning("Time window too small for pattern + chargebacks, adjusting start date")
                latest_start = 0
            pattern_start = random.randint(0, latest_start)
            pattern_start = start_date + timedelta(seconds=pattern_start)
            
            # Select merchants for all transactions
            merchant_sequence = super()._select_merchants(
                num_transactions,
                merchants,
                self.config['merchant_reuse_prob']
            )
            
            # Generate timestamps for all transactions
            timestamps = self._distribute_timestamps(num_transactions, pattern_start, window_max)
            
            # Create all transactions as fraudulent (all will get chargebacks)
            for i in range(num_transactions):
                timestamp = timestamps[i]
                merchant_id = merchant_sequence[i]
                
                # Generate transaction amount using global configuration range
                amount = round(random.uniform(global_amount_min, global_amount_max), 2)
                
                # Create transaction
                tx_id = str(uuid.uuid4())
                graph.add_node(
                    tx_id,
                    node_type='transaction',
                    timestamp=timestamp,
                    amount=amount,
                    is_fraudulent=True,
                    is_chargeback=False,
                    customer_id=customer_id
                )
                
                # Add edges
                graph.add_edge(card_id, tx_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, tx_id, edge_type='merchant_to_transaction')
                
                # Generate chargeback based on probability (not 100% anymore)
                if random.random() < self.config.get('chargeback_probability', 1.0):
                    cb_delay = random.randint(cb_delay_min, cb_delay_max)
                    cb_timestamp = timestamp + timedelta(days=cb_delay)
                    
                    # Create chargeback transaction
                    cb_id = str(uuid.uuid4())
                    graph.add_node(
                        cb_id,
                        node_type='transaction',
                        timestamp=cb_timestamp,
                        amount=amount,
                        is_fraudulent=True,
                        is_chargeback=True,
                        original_transaction=tx_id,
                        chargeback_reason="Fraudulent Transaction",
                        customer_id=customer_id
                    )
                    
                    # Add edges (same as original transaction)
                    graph.add_edge(card_id, cb_id, edge_type='card_to_transaction')
                    graph.add_edge(merchant_id, cb_id, edge_type='merchant_to_transaction')
        
        return graph 