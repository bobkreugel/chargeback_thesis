from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Set
import networkx as nx
import logging
from .base_pattern import BasePattern

logger = logging.getLogger(__name__)

class FriendlyFraudPattern(BasePattern):
    """
    Pattern generator for friendly fraud (initially legitimate customers who later commit chargeback fraud).
    
    Required config keys:
    - legitimate_period: {'min': int, 'max': int}  # Days of legitimate activity
    - initial_legitimate_transactions: {'min': int, 'max': int}  # Number of legitimate purchases
    - fraudulent_transactions: {'min': int, 'max': int}  # Number of fraudulent purchases
    - transaction_amount: {'min': float, 'max': float}  # Amount range in currency
    - time_between_transactions: {'min': int, 'max': int}  # Hours between transactions
    - chargeback_probability: float between 0 and 1
    - chargeback_delay: {'min': int, 'max': int}  # Days until chargeback
    - merchant_reuse_prob: float between 0 and 1
    """
    
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
        Inject friendly fraud patterns into the transaction graph.
        """
        if not merchants:
            raise ValueError("No merchants available for transactions")
            
        logger.info(f"Generating {num_patterns} friendly fraud patterns")
        
        # Generate patterns
        for pattern_idx in range(num_patterns):
            # Create customer with realistic data (appears legitimate initially)
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_fraudster_data()
            
            # Add customer node with all attributes
            graph.add_node(
                customer_id,
                node_type='customer',
                fraud_type='friendly_fraud',
                **customer_data
            )
            
            # Generate single card for this customer (starts as legitimate)
            card_id = str(uuid.uuid4())
            card_type = random.choice(['visa', 'mastercard', 'amex'])
            graph.add_node(
                card_id,
                node_type='card',
                card_type=card_type,
                is_fraudulent=False  # Card starts as legitimate
            )
            graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
            
            # Calculate pattern timeframe
            legitimate_period = random.randint(
                self.config['legitimate_period']['min'],
                self.config['legitimate_period']['max']
            )
            
            # Ensure pattern fits within overall timeframe
            latest_start = end_date - timedelta(
                days=legitimate_period + self.config['chargeback_delay']['max']
            )
            if latest_start < start_date:
                logger.warning(
                    f"Pattern {pattern_idx}: Legitimate period + chargeback delay exceeds available time window"
                )
                latest_start = start_date
            
            pattern_start = start_date + timedelta(
                seconds=random.randint(0, int((latest_start - start_date).total_seconds()))
            )
            legitimate_end = pattern_start + timedelta(days=legitimate_period)
            
            if legitimate_end > end_date:
                logger.warning(
                    f"Pattern {pattern_idx}: Legitimate period for customer {customer_id} extends beyond end_date"
                )
            
            # Phase 1: Generate initial legitimate transactions
            num_legitimate = random.randint(
                self.config['initial_legitimate_transactions']['min'],
                self.config['initial_legitimate_transactions']['max']
            )
            
            # Distribute legitimate transactions across the entire legitimate period
            # instead of clustering them with hours between
            legitimate_transaction_times = []
            
            # Ensure transactions are spread across the full legitimate period
            if num_legitimate > 1:
                # Calculate intervals to ensure good distribution across the entire period
                # Use 98% of the legitimate period to maximize spread while avoiding edge issues
                usable_period = legitimate_period * 0.98
                interval_days = usable_period / (num_legitimate - 1)
                
                for i in range(num_legitimate):
                    # Base time distribution
                    base_time = i * interval_days
                    # Add minimal random variation (max 5% of interval) to avoid perfect spacing
                    variation = random.uniform(-interval_days * 0.05, interval_days * 0.05)
                    
                    # Ensure we don't go negative or exceed our bounds
                    time_within_period = max(0, min(usable_period, base_time + variation))
                    transaction_time = pattern_start + timedelta(days=time_within_period)
                    
                    legitimate_transaction_times.append(transaction_time)
            else:
                # Single transaction: place it randomly within most of the legitimate period
                time_within_period = random.uniform(0, legitimate_period * 0.95)
                transaction_time = pattern_start + timedelta(days=time_within_period)
                legitimate_transaction_times.append(transaction_time)
            
            # Sort the transaction times to maintain chronological order
            legitimate_transaction_times.sort()
            
            previous_merchant = None
            used_merchants = set()  # Track all merchants used in this pattern
            
            logger.debug(
                f"Pattern {pattern_idx}: Starting legitimate phase with {num_legitimate} transactions over {legitimate_period} days"
            )
            
            for i, transaction_time in enumerate(legitimate_transaction_times):
                # Select merchant (with proper reuse tracking)
                merchant_id = super()._select_single_merchant(
                    merchants, self.config['merchant_reuse_prob'], previous_merchant, used_merchants
                )
                used_merchants.add(merchant_id)  # Track this merchant as used
                previous_merchant = merchant_id
                
                # Generate transaction amount
                amount = round(
                    random.uniform(
                        self.config['transaction_amount']['min'],
                        self.config['transaction_amount']['max']
                    ),
                    2
                )
                
                if transaction_time > end_date:
                    logger.warning(
                        f"Pattern {pattern_idx}: Transaction time {transaction_time} exceeds end_date"
                    )
                    break
                
                # Create legitimate transaction
                transaction_id = str(uuid.uuid4())
                graph.add_node(
                    transaction_id,
                    node_type='transaction',
                    amount=amount,
                    timestamp=transaction_time,
                    is_chargeback=False,
                    is_fraudulent=False,  # Explicitly mark as legitimate
                    has_chargeback=False,  # Flag for tracking chargebacks
                    customer_id=customer_id
                )
                
                # Add transaction edges
                graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
            
            # Continue with previous merchant for fraudulent phase (don't reset)
            # previous_merchant = None  # Remove this reset to maintain continuity
            
            # Phase 2: Generate fraudulent transactions after legitimate period
            num_fraudulent = random.randint(
                self.config['fraudulent_transactions']['min'],
                self.config['fraudulent_transactions']['max']
            )
            
            logger.debug(
                f"Pattern {pattern_idx}: Starting fraudulent phase with {num_fraudulent} transactions"
            )
            
            # Mark card as fraudulent at start of fraudulent phase
            graph.nodes[card_id]['is_fraudulent'] = True
            
            current_time = legitimate_end
            first_fraud = True
            
            for _ in range(num_fraudulent):
                # Select merchant (with proper reuse tracking)
                merchant_id = super()._select_single_merchant(
                    merchants, self.config['merchant_reuse_prob'], previous_merchant, used_merchants
                )
                used_merchants.add(merchant_id)  # Track this merchant as used
                previous_merchant = merchant_id
                
                # Generate transaction amount
                amount = round(
                    random.uniform(
                        self.config['transaction_amount']['min'],
                        self.config['transaction_amount']['max']
                    ),
                    2
                )
                
                # Add time between transactions (in hours)
                time_gap = random.randint(
                    self.config['time_between_transactions']['min'],
                    self.config['time_between_transactions']['max']
                )
                current_time += timedelta(hours=time_gap)
                
                if current_time > end_date:
                    logger.warning(
                        f"Pattern {pattern_idx}: Transaction time {current_time} exceeds end_date"
                    )
                    break
                
                # Create fraudulent transaction
                transaction_id = str(uuid.uuid4())
                graph.add_node(
                    transaction_id,
                    node_type='transaction',
                    amount=amount,
                    timestamp=current_time,
                    is_chargeback=False,
                    is_fraudulent=True,  # Mark as fraudulent transaction
                    has_chargeback=False,
                    customer_id=customer_id
                )
                
                # Add transaction edges
                graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
                
                # Add chargeback with configured probability
                if random.random() < self.config['chargeback_probability']:
                    # Try up to 3 times to generate a valid chargeback within end_date
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        # Calculate chargeback delay with random hours
                        # For each retry, reduce the delay range to increase success chance
                        delay_range_reduction = (attempt / max_attempts) * (
                            self.config['chargeback_delay']['max'] - 
                            self.config['chargeback_delay']['min']
                        )
                        max_delay = self.config['chargeback_delay']['max'] - delay_range_reduction
                        
                        delay_days = random.randint(
                            self.config['chargeback_delay']['min'],
                            int(max_delay)
                        )
                        delay_hours = random.randint(0, 23)
                        delay = timedelta(days=delay_days, hours=delay_hours)
                        
                        chargeback_time = current_time + delay
                        if chargeback_time <= end_date:
                            # Successfully found a valid chargeback time
                            chargeback_id = str(uuid.uuid4())
                            graph.add_node(
                                chargeback_id,
                                node_type='transaction',
                                amount=amount,
                                timestamp=chargeback_time,
                                is_chargeback=True,
                                is_fraudulent=True,
                                original_transaction=transaction_id,
                                customer_id=customer_id,
                                attempt_number=attempt + 1  # Track which attempt succeeded
                            )
                            
                            # Mark original transaction as having a chargeback
                            graph.nodes[transaction_id]['has_chargeback'] = True
                            
                            # Add chargeback edges
                            graph.add_edge(card_id, chargeback_id, edge_type='card_to_transaction')
                            graph.add_edge(merchant_id, chargeback_id, edge_type='merchant_to_transaction')
                            break
                    else:
                        # If all attempts failed, log it
                        logger.warning(
                            f"Pattern {pattern_idx}: Failed to generate chargeback within end_date "
                            f"after {max_attempts} attempts for transaction {transaction_id}"
                        )
            
            logger.debug(
                f"Pattern {pattern_idx}: Completed customer {customer_id} with "
                f"{num_legitimate} legitimate and {num_fraudulent} fraudulent transactions"
            )
        
        return graph
    

