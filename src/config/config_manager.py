import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages loading and validation of configuration settings for the fraud simulation."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Configuration Manager.
        
        Args:
            config_path (str, optional): Path to the configuration file. 
                                       If None, uses default_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'default_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the configuration settings."""
        if 'population' in self.config:
            self._validate_population_settings()
        
        if 'transactions' in self.config:
            self._validate_transaction_settings()
        
        if 'fraud_patterns' in self.config:
            self._validate_fraud_patterns()
        
        required_sections = {'population', 'transactions', 'fraud_patterns'}
        missing_sections = required_sections - set(self.config.keys())
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    def _validate_population_settings(self) -> None:
        """Validate population-related settings."""
        pop = self.config['population']
        
        if not pop.get('num_customers', 0) > 0:
            raise ValueError("population.num_customers must be greater than 0")
        
        if not pop.get('num_merchants', 0) > 0:
            raise ValueError("population.num_merchants must be greater than 0")
        
        cards = pop.get('cards_per_customer', {})
        if not (0 < cards.get('min', 0) <= cards.get('max', 0)):
            raise ValueError("Invalid cards_per_customer range")
    
    def _validate_transaction_settings(self) -> None:
        """Validate transaction-related settings."""
        trans = self.config['transactions']
        
        if not trans.get('num_transactions', 0) > 0:
            raise ValueError("transactions.num_transactions must be greater than 0")
        
        amount = trans.get('amount', {})
        if not (0 < amount.get('min', 0) <= amount.get('max', 0)):
            raise ValueError("Invalid transaction amount range")
        
        # Validate date range
        date_range = trans.get('date_range', {})
        try:
            start_date = datetime.strptime(date_range.get('start', ''), '%Y-%m-%d')
            end_date = datetime.strptime(date_range.get('end', ''), '%Y-%m-%d')
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            raise ValueError(f"Invalid date format in configuration: {str(e)}")
        
        # Validate legitimate chargebacks
        legit_cb = trans.get('legitimate_chargebacks', {})
        if not 0 <= legit_cb.get('rate', 0) <= 1:
            raise ValueError("legitimate_chargebacks.rate must be between 0 and 1")
        
        # Validate chargeback delay
        delay = legit_cb.get('delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid legitimate_chargebacks.delay range")
    
    def _validate_fraud_patterns(self) -> None:
        """Validate fraud pattern settings."""
        patterns = self.config['fraud_patterns']
        
        # Validate total fraud ratio
        if not 0 <= patterns.get('total_fraud_ratio', 0) <= 1:
            raise ValueError("fraud_patterns.total_fraud_ratio must be between 0 and 1")
            
        # Validate pattern distribution
        distribution = patterns.get('pattern_distribution', {})
        total = (
            distribution.get('bin_attack', 0) + 
            distribution.get('serial_chargeback', 0) + 
            distribution.get('friendly_fraud', 0) + 
            distribution.get('geographic_mismatch', 0)
        )
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Pattern distribution must sum to 1.0 (bin_attack + serial_chargeback + friendly_fraud + geographic_mismatch)")
            
        # Validate BIN Attack settings
        bin_attack = patterns.get('bin_attack', {})
        
        # Validate BIN prefixes
        bin_prefixes = bin_attack.get('bin_prefixes', [])
        if not bin_prefixes:
            raise ValueError("bin_attack.bin_prefixes must not be empty")
        for bin_prefix in bin_prefixes:
            if not (isinstance(bin_prefix, str) and len(bin_prefix) == 6 and bin_prefix.isdigit()):
                raise ValueError("Each BIN prefix must be a 6-digit string")
        
        # Validate number of cards range
        num_cards = bin_attack.get('num_cards', {})
        if not (0 < num_cards.get('min', 0) <= num_cards.get('max', 0)):
            raise ValueError("Invalid num_cards range in bin_attack")
        
        # Validate transaction amount range
        amount = bin_attack.get('transaction_amount', {})
        if not (0 < amount.get('min', 0) <= amount.get('max', 0)):
            raise ValueError("Invalid transaction_amount range in bin_attack")
            
        # Validate time window
        time_window = bin_attack.get('time_window', {})
        if not time_window.get('minutes', 0) > 0:
            raise ValueError("bin_attack.time_window.minutes must be greater than 0")
        
        # Validate merchant reuse probability
        if not 0 <= bin_attack.get('merchant_reuse_prob', 0) <= 1:
            raise ValueError("bin_attack.merchant_reuse_prob must be between 0 and 1")
            
        # Validate chargeback probability
        if not 0 <= bin_attack.get('chargeback_probability', 0) <= 1:
            raise ValueError("bin_attack.chargeback_probability must be between 0 and 1")
            
        # Validate chargeback delay
        delay = bin_attack.get('chargeback_delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid chargeback_delay range in bin_attack")
            
        # Validate Serial Chargeback settings
        serial_cb = patterns.get('serial_chargeback', {})
        
        # Validate transactions in pattern range
        txs_in_pattern = serial_cb.get('transactions_in_pattern', {})
        if not (0 < txs_in_pattern.get('min', 0) <= txs_in_pattern.get('max', 0)):
            raise ValueError("Invalid transactions_in_pattern range in serial_chargeback")
            
        # Validate time window
        time_window = serial_cb.get('time_window', {})
        if not (0 < time_window.get('min', 0) <= time_window.get('max', 0)):
            raise ValueError("Invalid time_window range in serial_chargeback")
            
        # Validate merchant reuse probability
        if not 0 <= serial_cb.get('merchant_reuse_prob', 0) <= 1:
            raise ValueError("serial_chargeback.merchant_reuse_prob must be between 0 and 1")
            
        # Validate chargeback delay
        delay = serial_cb.get('chargeback_delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid chargeback_delay range in serial_chargeback")
            
        # Validate Friendly Fraud settings
        friendly_fraud = patterns.get('friendly_fraud', {})
        
        # Validate legitimate period range
        legitimate_period = friendly_fraud.get('legitimate_period', {})
        if not (0 < legitimate_period.get('min', 0) <= legitimate_period.get('max', 0)):
            raise ValueError("Invalid legitimate_period range in friendly_fraud")
            
        # Validate initial legitimate transactions range
        initial_txs = friendly_fraud.get('initial_legitimate_transactions', {})
        if not (0 < initial_txs.get('min', 0) <= initial_txs.get('max', 0)):
            raise ValueError("Invalid initial_legitimate_transactions range in friendly_fraud")
            
        # Validate fraudulent transactions range
        fraud_txs = friendly_fraud.get('fraudulent_transactions', {})
        if not (0 < fraud_txs.get('min', 0) <= fraud_txs.get('max', 0)):
            raise ValueError("Invalid fraudulent_transactions range in friendly_fraud")
            
        # Validate time between transactions range
        time_between = friendly_fraud.get('time_between_transactions', {})
        if not (0 < time_between.get('min', 0) <= time_between.get('max', 0)):
            raise ValueError("Invalid time_between_transactions range in friendly_fraud")
            
        # Validate transaction amount range
        amount = friendly_fraud.get('transaction_amount', {})
        if not (0 < amount.get('min', 0) <= amount.get('max', 0)):
            raise ValueError("Invalid transaction_amount range in friendly_fraud")
            
        # Validate chargeback delay range
        delay = friendly_fraud.get('chargeback_delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid chargeback_delay range in friendly_fraud")
            
        # Validate chargeback probability
        if not 0 <= friendly_fraud.get('chargeback_probability', 0) <= 1:
            raise ValueError("friendly_fraud.chargeback_probability must be between 0 and 1")
            
        # Validate merchant reuse probability
        if not 0 <= friendly_fraud.get('merchant_reuse_prob', 0) <= 1:
            raise ValueError("friendly_fraud.merchant_reuse_prob must be between 0 and 1")
            
        # Validate Geographic Mismatch settings
        geographic_mismatch = patterns.get('geographic_mismatch', {})
        
        # Validate number of transactions range
        num_transactions = geographic_mismatch.get('num_transactions', {})
        if not (0 < num_transactions.get('min', 0) <= num_transactions.get('max', 0)):
            raise ValueError("Invalid num_transactions range in geographic_mismatch")
            
        # Validate time window range
        time_window = geographic_mismatch.get('time_window', {})
        if not (0 < time_window.get('min', 0) <= time_window.get('max', 0)):
            raise ValueError("Invalid time_window range in geographic_mismatch")
            
        # Validate transaction amount range
        amount = geographic_mismatch.get('transaction_amount', {})
        if not (0 < amount.get('min', 0) <= amount.get('max', 0)):
            raise ValueError("Invalid transaction_amount range in geographic_mismatch")
            
        # Validate merchant reuse probability
        if not 0 <= geographic_mismatch.get('merchant_reuse_prob', 0) <= 1:
            raise ValueError("geographic_mismatch.merchant_reuse_prob must be between 0 and 1")
            
        # Validate chargeback probability
        if not 0 <= geographic_mismatch.get('chargeback_probability', 0) <= 1:
            raise ValueError("geographic_mismatch.chargeback_probability must be between 0 and 1")
            
        # Validate chargeback delay range
        delay = geographic_mismatch.get('chargeback_delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid chargeback_delay range in geographic_mismatch")
    

    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Dict[str, Any]: The complete configuration
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section (str): The section name to retrieve
            
        Returns:
            Dict[str, Any]: The requested configuration section
            
        Raises:
            KeyError: If the section doesn't exist
        """
        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")
        return self.config[section] 