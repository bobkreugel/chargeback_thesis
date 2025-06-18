from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List, Set, Tuple
import networkx as nx
import logging
from faker import Faker
from .base_pattern import BasePattern

logger = logging.getLogger(__name__)

class GeographicMismatchPattern(BasePattern):
    """
    Pattern generator for geographic mismatch fraud.
    
    This pattern creates transactions where the billing address, shipping address, 
    and IP address are from different geographic locations, indicating potential fraud.
    
    Required config keys:
    - num_transactions: {'min': int, 'max': int}  # Number of transactions per pattern
    - transaction_amount: {'min': float, 'max': float}  # Amount range in currency
    - time_window: {'min': int, 'max': int}  # Days between first and last transaction
    - chargeback_probability: float between 0 and 1
    - chargeback_delay: {'min': int, 'max': int}  # Days until chargeback
    - merchant_reuse_prob: float between 0 and 1
    - geographic_regions: List of regions for generating diverse locations
    """
    
    # Geographic regions with countries for generating diverse locations
    GEOGRAPHIC_REGIONS = {
        'north_america': ['US', 'CA', 'MX'],
        'europe': ['NL', 'DE', 'FR', 'GB', 'IT', 'ES'],
        'asia': ['JP', 'CN', 'KR', 'IN', 'SG'],
        'africa': ['ZA', 'NG', 'EG', 'KE'],
        'south_america': ['BR', 'AR', 'CL', 'CO'],
        'oceania': ['AU', 'NZ']
    }
    
    # IP address ranges by region (simplified)
    IP_RANGES = {
        'north_america': ['192.168.0.0/16', '10.0.0.0/8'],
        'europe': ['172.16.0.0/12', '85.0.0.0/8'],
        'asia': ['203.0.0.0/8', '202.0.0.0/8'],
        'africa': ['196.0.0.0/8', '197.0.0.0/8'],
        'south_america': ['200.0.0.0/8', '201.0.0.0/8'],
        'oceania': ['210.0.0.0/8', '211.0.0.0/8']
    }
    
    def __init__(self, config: Dict[str, Any], seed: int = None):
        super().__init__(config, seed)
        
        # Create cached Faker instances for performance
        self._faker_cache = {}
        locale_map = {
            'US': 'en_US', 'CA': 'en_CA', 
            'NL': 'nl_NL', 'DE': 'de_DE', 'FR': 'fr_FR', 
            'GB': 'en_GB', 'IT': 'it_IT', 'ES': 'es_ES',
            'JP': 'ja_JP', 
            'BR': 'pt_BR', 'AU': 'en_AU'
        }
        
        # Pre-create Faker instances for all supported locales
        for country, locale in locale_map.items():
            try:
                self._faker_cache[country] = Faker(locale)
            except:
                self._faker_cache[country] = Faker('en_US')
        
        # Default faker for unsupported countries
        self._default_faker = Faker('en_US')

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
        Inject geographic mismatch patterns into the transaction graph.
        """
        if not merchants:
            raise ValueError("No merchants available for transactions")
            
        logger.info(f"Generating {num_patterns} geographic mismatch patterns")
        
        # Generate patterns
        for pattern_idx in range(num_patterns):
            # Create customer with realistic data
            customer_id = str(uuid.uuid4())
            customer_data = self._generate_fraudster_data()
            
            # Set customer to Netherlands as baseline
            customer_data['country'] = 'NL'
            customer_data['city'] = 'Amsterdam'
            customer_data['state'] = 'North Holland'
            
            # Add customer node with all attributes
            graph.add_node(
                customer_id,
                node_type='customer',
                fraud_type='geographic_mismatch',
                **customer_data
            )
            
            # Generate single card for this customer
            card_id = str(uuid.uuid4())
            card_type = random.choice(['visa', 'mastercard', 'amex'])
            graph.add_node(
                card_id,
                node_type='card',
                card_type=card_type,
                is_fraudulent=True
            )
            graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
            
            # Calculate number of transactions
            num_transactions = random.randint(
                self.config['num_transactions']['min'],
                self.config['num_transactions']['max']
            )
            
            # Calculate pattern timeframe
            time_window = random.randint(
                self.config['time_window']['min'],
                self.config['time_window']['max']
            )
            
            # Ensure pattern fits within overall timeframe
            latest_start = end_date - timedelta(
                days=time_window + self.config['chargeback_delay']['max']
            )
            if latest_start < start_date:
                logger.warning(
                    f"Pattern {pattern_idx}: Time window + chargeback delay exceeds available time window"
                )
                latest_start = start_date
            
            pattern_start = start_date + timedelta(
                seconds=random.randint(0, int((latest_start - start_date).total_seconds()))
            )
            
            # Generate timestamps for transactions
            transaction_times = self._generate_transaction_times(
                num_transactions, pattern_start, time_window
            )
            
            # Select merchants for transactions
            merchant_sequence = super()._select_merchants(
                num_transactions, merchants, self.config['merchant_reuse_prob']
            )
            
            # Generate transactions with geographic mismatches
            for i, transaction_time in enumerate(transaction_times):
                merchant_id = merchant_sequence[i]
                
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
                
                # Create transaction
                transaction_id = str(uuid.uuid4())
                graph.add_node(
                    transaction_id,
                    node_type='transaction',
                    amount=amount,
                    timestamp=transaction_time,
                    is_chargeback=False,
                    is_fraudulent=True,
                    customer_id=customer_id
                )
                
                # Add basic transaction edges
                graph.add_edge(card_id, transaction_id, edge_type='card_to_transaction')
                graph.add_edge(merchant_id, transaction_id, edge_type='merchant_to_transaction')
                
                # Create geographic mismatch locations
                billing_addr, shipping_addr, ip_location = self._generate_mismatched_locations()
                
                # Create billing address node
                billing_id = f"billing_{transaction_id}"
                graph.add_node(
                    billing_id,
                    node_type='location',
                    location_type='billing_address',
                    address=billing_addr['address'],
                    city=billing_addr['city'],
                    state=billing_addr['state'],
                    country=billing_addr['country'],
                    zip_code=billing_addr['zip'],
                    region=billing_addr['region']
                )
                graph.add_edge(billing_id, transaction_id, edge_type='billing_to_transaction')
                
                # Create shipping address node  
                shipping_id = f"shipping_{transaction_id}"
                graph.add_node(
                    shipping_id,
                    node_type='location',
                    location_type='shipping_address',
                    address=shipping_addr['address'],
                    city=shipping_addr['city'],
                    state=shipping_addr['state'],
                    country=shipping_addr['country'],
                    zip_code=shipping_addr['zip'],
                    region=shipping_addr['region']
                )
                graph.add_edge(shipping_id, transaction_id, edge_type='shipping_to_transaction')
                
                # Create IP address node
                ip_id = f"ip_{transaction_id}"
                graph.add_node(
                    ip_id,
                    node_type='location',
                    location_type='ip_address',
                    ip_address=ip_location['ip'],
                    country=ip_location['country'],
                    region=ip_location['region'],
                    is_fraudulent=True
                )
                graph.add_edge(ip_id, transaction_id, edge_type='ip_to_transaction')
                
                # Add chargeback with configured probability
                if random.random() < self.config['chargeback_probability']:
                    delay = random.randint(
                        self.config['chargeback_delay']['min'],
                        self.config['chargeback_delay']['max']
                    )
                    
                    chargeback_id = str(uuid.uuid4())
                    chargeback_timestamp = transaction_time + timedelta(days=delay)
                    
                    graph.add_node(
                        chargeback_id,
                        node_type='transaction',
                        amount=amount,
                        timestamp=chargeback_timestamp,
                        is_chargeback=True,
                        is_fraudulent=True,
                        original_transaction=transaction_id,
                        customer_id=customer_id
                    )
                    
                    # Add same edges as original transaction
                    graph.add_edge(card_id, chargeback_id, edge_type='card_to_transaction')
                    graph.add_edge(merchant_id, chargeback_id, edge_type='merchant_to_transaction')
                    graph.add_edge(billing_id, chargeback_id, edge_type='billing_to_transaction')
                    graph.add_edge(shipping_id, chargeback_id, edge_type='shipping_to_transaction')
                    graph.add_edge(ip_id, chargeback_id, edge_type='ip_to_transaction')
        
        return graph
    
    def _generate_transaction_times(
        self, 
        num_transactions: int, 
        pattern_start: datetime, 
        time_window_days: int
    ) -> List[datetime]:
        """Generate transaction timestamps distributed across the time window."""
        times = []
        
        if num_transactions == 1:
            # Single transaction: place it randomly in the first half of the window
            offset_hours = random.randint(0, time_window_days * 12)  # First half of window
            times.append(pattern_start + timedelta(hours=offset_hours))
        else:
            # Multiple transactions: distribute across the time window
            for i in range(num_transactions):
                # Distribute evenly with some randomness
                base_offset = (time_window_days * 24 * i) / (num_transactions - 1) if num_transactions > 1 else 0
                random_offset = random.uniform(-12, 12)  # +/- 12 hours variation
                total_offset = max(0, base_offset + random_offset)
                
                transaction_time = pattern_start + timedelta(hours=total_offset)
                times.append(transaction_time)
        
        return sorted(times)
    

    
    def _generate_mismatched_locations(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Generate billing address, shipping address, and IP location from different regions.
        This creates the geographic mismatch that indicates potential fraud.
        """
        # Select three different regions
        regions = list(self.GEOGRAPHIC_REGIONS.keys())
        selected_regions = random.sample(regions, min(3, len(regions)))
        
        # Ensure we have at least 3 regions (repeat if necessary)
        while len(selected_regions) < 3:
            selected_regions.append(random.choice(regions))
        
        billing_region = selected_regions[0]
        shipping_region = selected_regions[1]
        ip_region = selected_regions[2]
        
        # Generate billing address (customer's region - Netherlands/Europe)
        billing_addr = self._generate_address_for_region('europe')
        billing_addr['region'] = 'europe'
        
        # Generate shipping address (different continent)
        shipping_addr = self._generate_address_for_region(shipping_region)
        shipping_addr['region'] = shipping_region
        
        # Generate IP location (third different region)
        ip_location = self._generate_ip_for_region(ip_region)
        ip_location['region'] = ip_region
        
        return billing_addr, shipping_addr, ip_location
    
    def _generate_address_for_region(self, region: str) -> Dict[str, Any]:
        """Generate a realistic address for a specific geographic region."""
        countries = self.GEOGRAPHIC_REGIONS[region]
        country = random.choice(countries)
        
        # Use cached Faker instance for better performance
        fake_local = self._faker_cache.get(country, self._default_faker)
        
        try:
            state = fake_local.state() if country in ['US', 'CA', 'AU'] else fake_local.city()
        except AttributeError:
            state = fake_local.city()
        
        return {
            'address': fake_local.street_address(),
            'city': fake_local.city(),
            'state': state,
            'country': country,
            'zip': fake_local.postcode()
        }
    
    def _generate_ip_for_region(self, region: str) -> Dict[str, Any]:
        """Generate an IP address for a specific geographic region."""
        countries = self.GEOGRAPHIC_REGIONS[region]
        country = random.choice(countries)
        
        # Generate a realistic IP address (simplified approach)
        if region == 'north_america':
            ip = f"{random.randint(192, 199)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif region == 'europe':
            ip = f"{random.randint(85, 95)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif region == 'asia':
            ip = f"{random.randint(202, 210)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif region == 'africa':
            ip = f"{random.randint(196, 199)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif region == 'south_america':
            ip = f"{random.randint(200, 201)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:  # oceania
            ip = f"{random.randint(210, 220)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        return {
            'ip': ip,
            'country': country
        } 