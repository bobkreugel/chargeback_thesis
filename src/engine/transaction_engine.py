import networkx as nx
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, List, Tuple, Any
import logging
import pandas as pd
import os
import time
from src.config.config_manager import ConfigurationManager
from src.patterns.serial_chargeback import SerialChargebackPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionEngine:
    """
    Core engine for generating synthetic transaction data and maintaining the graph structure.
    Handles creation of base population and normal transactions.
    """
    
    # Define merchant categories
    MERCHANT_CATEGORIES = [
        'Retail', 'Electronics', 'Fashion', 'Food & Dining',
        'Travel', 'Entertainment', 'Home & Garden', 'Health & Beauty',
        'Sports & Outdoors', 'Digital Services', 'Education',
        'Automotive', 'Jewelry', 'Books & Media'
    ]
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the Transaction Engine.
        
        Args:
            config_manager: Configuration manager instance with validated settings
        """
        self.config = config_manager
        self.faker = Faker()
        self.graph = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple transactions between same nodes
        
        # Store entity mappings for quick access
        self.customers: Dict[str, Dict] = {}
        self.merchants: Dict[str, Dict] = {}
        self.cards: Dict[str, Dict] = {}
        
        # Initialize random seed based on current time
        current_seed = int(time.time())
        random.seed(current_seed)
        Faker.seed(current_seed)
        logger.info(f"Initialized with random seed: {current_seed}")
    
    def generate_base_population(self) -> None:
        """Generate the base population of customers, cards, and merchants."""
        logger.info("Generating base population...")
        
        pop_config = self.config.get_section('population')
        
        self._generate_customers(pop_config['num_customers'])
        self._generate_merchants(pop_config['num_merchants'])
        self._assign_cards_to_customers(pop_config['cards_per_customer'])
        
        logger.info(f"Generated {len(self.customers)} customers, {len(self.merchants)} merchants, "
                   f"and {len(self.cards)} cards")
    
    def _generate_customers(self, num_customers: int) -> None:
        """Generate customer nodes with realistic attributes."""
        for _ in range(num_customers):
            customer_id = str(uuid.uuid4())
            customer_data = {
                'id': customer_id,
                'name': self.faker.name(),
                'email': self.faker.email(),
                'phone': self.faker.phone_number(),
                'address': self.faker.address(),
                'created_at': self.faker.date_time_between(
                    start_date='-5y',
                    end_date='now'
                ).isoformat()
            }
            
            self.customers[customer_id] = customer_data
            self.graph.add_node(customer_id, **customer_data, node_type='customer')
    
    def _generate_merchants(self, num_merchants: int) -> None:
        """Generate merchant nodes with realistic attributes."""
        for _ in range(num_merchants):
            merchant_id = str(uuid.uuid4())
            merchant_data = {
                'id': merchant_id,
                'name': self.faker.company(),
                'category': random.choice(self.MERCHANT_CATEGORIES),
                'address': self.faker.address(),
                'created_at': self.faker.date_time_between(
                    start_date='-5y',
                    end_date='now'
                ).isoformat()
            }
            
            self.merchants[merchant_id] = merchant_data
            self.graph.add_node(merchant_id, **merchant_data, node_type='merchant')
    
    def _assign_cards_to_customers(self, cards_per_customer: Dict[str, int]) -> None:
        """Assign payment cards to customers."""
        for customer_id in self.customers:
            num_cards = random.randint(cards_per_customer['min'], cards_per_customer['max'])
            
            for _ in range(num_cards):
                card_id = str(uuid.uuid4())
                card_data = {
                    'id': card_id,
                    'number': self.faker.credit_card_number(card_type=None),
                    'type': random.choice(['visa', 'mastercard', 'amex']),
                    'expiry_date': self.faker.credit_card_expire(),
                    'created_at': self.faker.date_time_between(
                        start_date='-2y',
                        end_date='now'
                    ).isoformat()
                }
                
                self.cards[card_id] = card_data
                self.graph.add_node(card_id, **card_data, node_type='card')
                
                # Add edge from customer to card (ownership)
                self.graph.add_edge(
                    customer_id,
                    card_id,
                    relationship_type='HAS_CARD',
                    created_at=card_data['created_at']
                )
    
    def generate_normal_transactions(self) -> None:
        """Generate normal transaction patterns including legitimate chargebacks."""
        logger.info("Generating normal transactions...")
        
        trans_config = self.config.get_section('transactions')
        date_range = trans_config['date_range']
        start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
        end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
        
        total_transactions = trans_config['total_transactions']
        legitimate_cb_rate = trans_config['legitimate_chargebacks']['rate']
        
        transactions_generated = 0
        
        while transactions_generated < total_transactions:
            # Select random card and merchant
            card_id = random.choice(list(self.cards.keys()))
            merchant_id = random.choice(list(self.merchants.keys()))
            
            # Generate normal transaction
            transaction = self._create_transaction(
                card_id,
                merchant_id,
                start_date,
                end_date,
                trans_config['amount_range']
            )
            
            # Determine if this will be a legitimate chargeback
            is_chargeback = random.random() < legitimate_cb_rate
            if is_chargeback:
                self._add_legitimate_chargeback(transaction, trans_config['legitimate_chargebacks'])
            
            transactions_generated += 1
            
            if transactions_generated % 1000 == 0:
                logger.info(f"Generated {transactions_generated} transactions...")
        
        logger.info(f"Completed generating {total_transactions} transactions")
        
        # Inject patterns after normal transactions are generated
        self._inject_patterns(start_date, end_date, trans_config['amount_range'])
    
    def _inject_patterns(
        self,
        start_date: datetime,
        end_date: datetime,
        amount_range: Dict[str, float]
    ) -> None:
        """
        Inject configured patterns into the transaction graph.
        
        Args:
            start_date: Start of possible transaction dates
            end_date: End of possible transaction dates
            amount_range: Range for transaction amounts
        """
        logger.info("Injecting transaction patterns...")
        
        # Initialize and inject serial chargeback pattern
        pattern_config = self.config.get_section('pattern_injection')
        serial_cb_pattern = SerialChargebackPattern(pattern_config)
        
        self.graph = serial_cb_pattern.inject(
            self.graph,
            start_date,
            end_date,
            amount_range
        )
        
        logger.info("Completed pattern injection")
    
    def _create_transaction(
        self,
        card_id: str,
        merchant_id: str,
        start_date: datetime,
        end_date: datetime,
        amount_range: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create a single transaction with all necessary attributes."""
        transaction_id = str(uuid.uuid4())
        transaction_date = self.faker.date_time_between(
            start_date=start_date,
            end_date=end_date
        )
        
        transaction_data = {
            'id': transaction_id,
            'amount': round(random.uniform(amount_range['min'], amount_range['max']), 2),
            'currency': 'USD',
            'timestamp': transaction_date.isoformat(),
            'status': 'completed',
            'is_chargeback': False,
            'chargeback_reason': None,
            'chargeback_date': None
        }
        
        # Add transaction node
        self.graph.add_node(transaction_id, **transaction_data, node_type='transaction')
        
        # Add edges for the transaction
        self.graph.add_edge(
            card_id,
            transaction_id,
            relationship_type='MADE_TRANSACTION',
            timestamp=transaction_date.isoformat()
        )
        self.graph.add_edge(
            transaction_id,
            merchant_id,
            relationship_type='PAID_TO',
            timestamp=transaction_date.isoformat()
        )
        
        return transaction_data
    
    def _add_legitimate_chargeback(
        self,
        transaction: Dict[str, Any],
        chargeback_config: Dict[str, Any]
    ) -> None:
        """Add legitimate chargeback attributes to a transaction."""
        transaction_date = datetime.fromisoformat(transaction['timestamp'])
        delay_days = random.randint(
            chargeback_config['chargeback_delay']['min'],
            chargeback_config['chargeback_delay']['max']
        )
        
        reason = random.choices(
            list(chargeback_config['reasons'].keys()),
            weights=list(chargeback_config['reasons'].values())
        )[0]
        
        transaction['is_chargeback'] = True
        transaction['status'] = 'chargeback'
        transaction['chargeback_reason'] = reason
        transaction['chargeback_date'] = (transaction_date + timedelta(days=delay_days)).isoformat()
        
        # Update node in graph
        self.graph.nodes[transaction['id']].update(transaction)
    
    def get_graph(self) -> nx.MultiDiGraph:
        """Return the generated transaction graph."""
        return self.graph
    
    def export_to_csv(self, output_path: str) -> None:
        """
        Export transaction data to CSV format, including related customer and merchant information.
        
        Args:
            output_path (str): Directory where CSV files will be saved
        """
        logger.info("Exporting data to CSV...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare transaction data with related information
        transactions_data = []
        
        for node, attr in self.graph.nodes(data=True):
            if attr.get('node_type') == 'transaction':
                # Get card that made the transaction
                card_edges = list(self.graph.in_edges(node, data=True))
                card_id = card_edges[0][0]
                card_data = self.cards[card_id]
                
                # Get customer who owns the card
                customer_edges = list(self.graph.in_edges(card_id, data=True))
                customer_id = customer_edges[0][0]
                customer_data = self.customers[customer_id]
                
                # Get merchant who received the payment
                merchant_edges = list(self.graph.out_edges(node, data=True))
                merchant_id = merchant_edges[0][1]
                merchant_data = self.merchants[merchant_id]
                
                # Combine all data
                transaction_record = {
                    # Transaction information
                    'transaction_id': attr['id'],
                    'amount': attr['amount'],
                    'currency': attr['currency'],
                    'timestamp': attr['timestamp'],
                    'status': attr['status'],
                    'is_chargeback': attr['is_chargeback'],
                    'chargeback_reason': attr['chargeback_reason'],
                    'chargeback_date': attr['chargeback_date'],
                    
                    # Customer information
                    'customer_id': customer_data['id'],
                    'customer_name': customer_data['name'],
                    'customer_email': customer_data['email'],
                    'customer_phone': customer_data['phone'],
                    'customer_address': customer_data['address'],
                    'customer_created_at': customer_data['created_at'],
                    
                    # Card information
                    'card_id': card_data['id'],
                    'card_type': card_data['type'],
                    'card_number': card_data['number'],
                    'card_expiry': card_data['expiry_date'],
                    'card_created_at': card_data['created_at'],
                    
                    # Merchant information
                    'merchant_id': merchant_data['id'],
                    'merchant_name': merchant_data['name'],
                    'merchant_category': merchant_data['category'],
                    'merchant_address': merchant_data['address'],
                    'merchant_created_at': merchant_data['created_at']
                }
                
                transactions_data.append(transaction_record)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(transactions_data)
        
        # Save main transaction file
        transactions_file = os.path.join(output_path, 'transactions.csv')
        df.to_csv(transactions_file, index=False)
        
        # Also save separate entity files for convenience
        customers_df = pd.DataFrame([data for data in self.customers.values()])
        merchants_df = pd.DataFrame([data for data in self.merchants.values()])
        cards_df = pd.DataFrame([data for data in self.cards.values()])
        
        customers_df.to_csv(os.path.join(output_path, 'customers.csv'), index=False)
        merchants_df.to_csv(os.path.join(output_path, 'merchants.csv'), index=False)
        cards_df.to_csv(os.path.join(output_path, 'cards.csv'), index=False)
        
        logger.info(f"Exported {len(transactions_data)} transactions to {output_path}")
        logger.info(f"Created files: transactions.csv, customers.csv, merchants.csv, cards.csv") 