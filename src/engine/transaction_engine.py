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
from src.patterns.bin_attack import BINAttackPattern
from collections import defaultdict
import sys
import csv

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
    
    def __init__(self, config_manager, seed: int = None):
        """Initialize the Transaction Engine."""
        self.config = config_manager
        self.graph = nx.DiGraph()
        self.customers = {}  # customer_id -> customer data
        self.merchants = {}  # merchant_id -> merchant data
        self.cards = {}  # card_id -> customer_id mapping
        self.customer_cards = defaultdict(set)  # customer_id -> set of card_ids
        
        # Initialize random seeds if provided
        self.seed = seed
        if seed is not None:
            self.fake = Faker()
            Faker.seed(seed)
            random.seed(seed)
            logger.info(f"Initialized with fixed random seed: {seed}")
        else:
            self.fake = Faker()
            random.seed()
            logger.info("Initialized with random seed")
        
        # Initialize date range
        trans_config = self.config.get_section('transactions')
        date_range = trans_config['date_range']
        self.start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
        self.end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
        
    def _generate_customer_data(self) -> Dict[str, Any]:
        """Generate realistic customer data using Faker."""
        return {
            'name': self.fake.name(),
            'email': self.fake.email(),
            'phone': self.fake.phone_number(),
            'address': self.fake.address().replace('\n', ', '),
            'city': self.fake.city(),
            'state': self.fake.state(),
            'zip': self.fake.zipcode(),
            'risk_score': round(random.uniform(0, 100), 2)
        }

    def _generate_merchant_data(self) -> Dict[str, Any]:
        """Generate realistic merchant data using Faker."""
        category = random.choice(self.MERCHANT_CATEGORIES)
        return {
            'name': self.fake.company(),
            'category': category,
            'address': self.fake.address().replace('\n', ', '),
            'city': self.fake.city(),
            'state': self.fake.state(),
            'zip': self.fake.zipcode(),
            'phone': self.fake.phone_number(),
            'website': self.fake.url()
        }

    def _generate_timestamp(self) -> datetime:
        """Generate a random timestamp within the configured date range."""
        time_between_dates = self.end_date - self.start_date
        days_between = time_between_dates.days
        random_days = random.randrange(days_between)
        random_seconds = random.randrange(24 * 60 * 60)  # Random time within the day
        return self.start_date + timedelta(days=random_days, seconds=random_seconds)
        
    def _create_transaction(self, card_id: str, merchant_id: str, amount: float, timestamp: datetime) -> str:
        """Create a transaction node in the graph."""
        transaction_id = str(uuid.uuid4())
        
        # Add transaction node
        self.graph.add_node(
            transaction_id,
            node_type='transaction',
            amount=amount,
            timestamp=timestamp,
            is_chargeback=False
        )
        
        # Add edges
        self.graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
        self.graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
        
        return transaction_id
        
    def _create_chargeback(self, transaction_id: str, delay_days: int) -> str:
        """Create a chargeback node for a transaction."""
        # Get original transaction details
        transaction = self.graph.nodes[transaction_id]
        
        # Create chargeback transaction
        chargeback_id = str(uuid.uuid4())
        chargeback_timestamp = transaction['timestamp'] + timedelta(days=delay_days)
        
        self.graph.add_node(
            chargeback_id,
            node_type='transaction',
            amount=transaction['amount'],
            timestamp=chargeback_timestamp,
            is_chargeback=True,
            original_transaction=transaction_id
        )
        
        # Add edges (same as original transaction)
        for edge in self.graph.in_edges(transaction_id):
            self.graph.add_edge(edge[0], chargeback_id, edge_type=self.graph.edges[edge]['edge_type'])
        
        return chargeback_id
    
    def generate_base_population(self) -> None:
        """Generate the base population of customers, merchants, and cards."""
        logger.info("Generating base population...")
        
        # Get configuration
        pop_config = self.config.get_section('population')
        num_customers = pop_config['num_customers']
        num_merchants = pop_config['num_merchants']
        cards_range = pop_config['cards_per_customer']
        
        # Generate customers with realistic data
        for _ in range(num_customers):
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_customer_data()
            self.customers[customer_id] = customer_data
            
            # Add customer node with all attributes
            self.graph.add_node(
                customer_id,
                node_type='customer',
                **customer_data
            )
            
            # Generate cards for customer
            num_cards = random.randint(cards_range['min'], cards_range['max'])
            for _ in range(num_cards):
                card_id = str(uuid.uuid4())
                self.cards[card_id] = customer_id
                self.customer_cards[customer_id].add(card_id)
                
                # Add card node with type info
                card_type = random.choice(['visa', 'mastercard', 'amex'])
                self.graph.add_node(
                    card_id,
                    node_type='card',
                    card_type=card_type
                )
                self.graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
        
        # Generate merchants with realistic data
        for _ in range(num_merchants):
            merchant_id = str(uuid.uuid4())
            merchant_data = self._generate_merchant_data()
            self.merchants[merchant_id] = merchant_data
            
            # Add merchant node with all attributes
            self.graph.add_node(
                merchant_id,
                node_type='merchant',
                **merchant_data
            )
        
        logger.info(f"Generated {num_customers} customers, {num_merchants} merchants, and {len(self.cards)} cards")
    
    def generate_normal_transactions(self) -> None:
        """Generate normal (non-fraudulent) transactions."""
        logger.info("Generating normal transactions...")
        
        trans_config = self.config.get_section('transactions')
        num_transactions = trans_config['num_transactions']
        amount_range = trans_config['amount']
        legit_cb = trans_config['legitimate_chargebacks']
        
        # Calculate number of normal transactions (excluding fraud)
        fraud_config = self.config.get_section('fraud_patterns')
        total_fraud = int(num_transactions * fraud_config['total_fraud_ratio'])
        normal_transactions = num_transactions - total_fraud
        
        logger.info(f"Generating {normal_transactions} normal transactions...")
        
        transactions_generated = 0
        while transactions_generated < normal_transactions:
            # Select random customer and their card
            customer_id = random.choice(list(self.customers))
            card_id = random.choice(list(self.customer_cards[customer_id]))
            
            # Select random merchant
            merchant_id = random.choice(list(self.merchants))
            
            # Generate random amount
            amount = round(random.uniform(amount_range['min'], amount_range['max']), 2)
            
            # Create transaction
            transaction_id = self._create_transaction(
                card_id=card_id,
                merchant_id=merchant_id,
                amount=amount,
                timestamp=self._generate_timestamp()
            )
            
            # Determine if this will be a legitimate chargeback
            if random.random() < legit_cb['rate']:
                delay = random.randint(legit_cb['delay']['min'], legit_cb['delay']['max'])
                self._create_chargeback(transaction_id, delay)
            
            transactions_generated += 1
            
            if transactions_generated % 1000 == 0:
                logger.info(f"Generated {transactions_generated} transactions...")
        
        logger.info(f"Completed generating {normal_transactions} normal transactions")
    
    def _refresh_internal_mappings(self) -> None:
        """Update internal customer and card mappings with new nodes from the graph."""
        for node, attr in self.graph.nodes(data=True):
            if attr.get('node_type') == 'customer' and node not in self.customers:
                # Add new customer data (without is_fraudster flag in dictionary)
                customer_data = {
                    'name': attr.get('name', ''),
                    'email': attr.get('email', ''),
                    'phone': attr.get('phone', ''),
                    'address': attr.get('address', ''),
                    'city': attr.get('city', ''),
                    'state': attr.get('state', ''),
                    'zip': attr.get('zip', ''),
                    'risk_score': attr.get('risk_score', 0)
                }
                self.customers[node] = customer_data
                
                # Keep is_fraudster in graph node
                if attr.get('is_fraudster'):
                    self.graph.nodes[node]['is_fraudster'] = True
                
            elif attr.get('node_type') == 'card' and node not in self.cards:
                # Find customer for this card
                customer_edges = [e for e in self.graph.in_edges(node, data=True) if e[2].get('edge_type') == 'customer_to_card']
                if customer_edges:
                    customer_id = customer_edges[0][0]
                    self.cards[node] = customer_id
                    self.customer_cards[customer_id].add(node)

    def _inject_patterns(self) -> None:
        """Inject configured fraud patterns into the transaction graph."""
        logger.info("Injecting fraud patterns...")
        
        # Calculate number of patterns based on fraud ratio
        fraud_config = self.config.get_section('fraud_patterns')
        total_transactions = self.config.get_section('transactions')['num_transactions']
        target_fraud = int(total_transactions * fraud_config['total_fraud_ratio'])
        
        logger.info(f"Target fraudulent transactions: {target_fraud}")
        
        # Calculate number of each pattern type
        pattern_dist = fraud_config['pattern_distribution']
        target_serial_cb = int(target_fraud * pattern_dist['serial_chargeback'])
        target_bin_attacks = int(target_fraud * pattern_dist['bin_attack'])
        
        # Calculate average transactions per pattern
        bin_config = fraud_config['bin_attack']
        avg_cards_per_bin = (bin_config['num_cards']['min'] + bin_config['num_cards']['max']) / 2
        avg_txs_per_bin = avg_cards_per_bin * (1 + bin_config['chargeback_rate'])  # Original + chargebacks
        
        serial_config = fraud_config['serial_chargeback']
        avg_txs_per_serial = (
            (serial_config['transactions_in_pattern']['min'] + serial_config['transactions_in_pattern']['max']) / 2
        ) * 2  # Each transaction gets a chargeback
        
        # Calculate number of patterns needed
        num_serial_cb = int(target_serial_cb / avg_txs_per_serial)
        num_bin_attacks = int(target_bin_attacks / avg_txs_per_bin)
        
        logger.info(f"Generating patterns:")
        logger.info(f"- Serial chargebacks: {num_serial_cb} patterns (avg {avg_txs_per_serial:.1f} txs/pattern)")
        logger.info(f"- BIN attacks: {num_bin_attacks} patterns (avg {avg_txs_per_bin:.1f} txs/pattern)")
        
        # Initialize pattern generators with same seed if one was provided
        serial_cb_pattern = SerialChargebackPattern(fraud_config['serial_chargeback'], seed=self.seed)
        bin_attack_pattern = BINAttackPattern(fraud_config['bin_attack'], seed=self.seed)
        
        # Get date range from config
        trans_config = self.config.get_section('transactions')
        date_range = trans_config['date_range']
        start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
        end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
        
        # Inject serial chargeback patterns
        logger.info(f"Generating {num_serial_cb} serial chargeback patterns")
        self.graph = serial_cb_pattern.inject(
            self.graph,
            num_serial_cb,
            start_date,
            end_date,
            self.customers,
            self.customer_cards,
            self.merchants
        )
        
        # Update internal mappings after serial chargebacks
        self._refresh_internal_mappings()
        
        # Inject BIN attack patterns
        logger.info(f"Generating {num_bin_attacks} BIN attack patterns")
        self.graph = bin_attack_pattern.inject(
            self.graph,
            num_bin_attacks,
            start_date,
            end_date,
            self.customers,
            self.customer_cards,
            self.merchants
        )
        
        # Update internal mappings after BIN attacks
        self._refresh_internal_mappings()
        
        logger.info("Completed fraud pattern injection")
    
    def get_graph(self) -> nx.MultiDiGraph:
        """Return the generated transaction graph."""
        return self.graph
    
    def export_to_csv(self, output_dir: str) -> None:
        """Export the generated data to CSV files."""
        logger.info("Exporting data to CSV...")
        
        # Export customers with all attributes
        customer_rows = []
        for customer_id, customer_data in self.customers.items():
            row = {'customer_id': customer_id}
            row.update(customer_data)  # Add all customer attributes
            row['num_cards'] = len(self.customer_cards[customer_id])
            customer_rows.append(row)
        
        self._write_csv(
            os.path.join(output_dir, 'customers.csv'),
            customer_rows
        )
        
        # Export merchants with all attributes
        merchant_rows = []
        for merchant_id, merchant_data in self.merchants.items():
            row = {'merchant_id': merchant_id}
            row.update(merchant_data)  # Add all merchant attributes
            merchant_rows.append(row)
        
        self._write_csv(
            os.path.join(output_dir, 'merchants.csv'),
            merchant_rows
        )
        
        # Export cards with customer mapping and type information
        card_rows = []
        for card_id, customer_id in self.cards.items():
            card_data = self.graph.nodes[card_id]
            row = {
                'card_id': card_id,
                'customer_id': customer_id,
                'card_type': card_data.get('card_type', 'unknown'),
                'is_fraudulent': card_data.get('is_fraudulent', False)
            }
            card_rows.append(row)
        
        self._write_csv(
            os.path.join(output_dir, 'cards.csv'),
            card_rows
        )
        
        # Export transactions with all relationships
        transaction_rows = []
        for node, attr in self.graph.nodes(data=True):
            if attr.get('node_type') == 'transaction':
                # Get card and merchant for this transaction
                card_edges = [e for e in self.graph.in_edges(node, data=True) if e[2].get('edge_type') == 'card_to_transaction']
                merchant_edges = [e for e in self.graph.in_edges(node, data=True) if e[2].get('edge_type') == 'merchant_to_transaction']
                ip_edges = [e for e in self.graph.in_edges(node, data=True) if e[2].get('edge_type') == 'ip_to_transaction']
                
                if card_edges and merchant_edges:
                    card_id = card_edges[0][0]
                    merchant_id = merchant_edges[0][0]
                    customer_id = self.cards[card_id]
                    
                    # Get customer and merchant data
                    customer_data = self.customers[customer_id]
                    merchant_data = self.merchants[merchant_id]
                    
                    # Get source IP if it exists
                    source_ip = None
                    if ip_edges:
                        ip_node_id = ip_edges[0][0]
                        source_ip = self.graph.nodes[ip_node_id].get('address')
                    
                    # Create transaction row with all relationships
                    row = {
                        'transaction_id': node,
                        'amount': attr.get('amount', 0.0),
                        'timestamp': attr.get('timestamp', ''),
                        'is_chargeback': attr.get('is_chargeback', False),
                        'chargeback_reason': attr.get('chargeback_reason', ''),
                        'chargeback_date': attr.get('chargeback_date', ''),
                        'original_transaction': attr.get('original_transaction', ''),
                        'source_ip': source_ip,  # Add source IP to transaction data
                        'card_id': card_id,
                        'customer_id': customer_id,
                        'customer_name': customer_data.get('name', ''),
                        'customer_email': customer_data.get('email', ''),
                        'merchant_id': merchant_id,
                        'merchant_name': merchant_data.get('name', ''),
                        'merchant_category': merchant_data.get('category', '')
                    }
                    transaction_rows.append(row)
        
        self._write_csv(
            os.path.join(output_dir, 'transactions.csv'),
            transaction_rows
        )
        
    def _write_csv(self, filepath: str, rows: List[Dict[str, Any]]) -> None:
        """Write rows to a CSV file with dynamic field names."""
        if not rows:
            logger.warning(f"No rows to write to {filepath}")
            return
            
        # Get all unique field names from all rows
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())
        fieldnames = sorted(list(fieldnames))  # Sort for consistency
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows) 