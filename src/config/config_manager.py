import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
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
        
        if 'pattern_injection' in self.config:
            self._validate_pattern_injection_settings()
        
        self._validate_probability_distributions()
        
        required_sections = {'population', 'transactions', 'pattern_injection'}
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
        
        if not trans.get('total_transactions', 0) > 0:
            raise ValueError("transactions.total_transactions must be greater than 0")
        
        amount = trans.get('amount_range', {})
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
        
        # Validate legitimate chargeback reasons probabilities sum to 1
        reasons = legit_cb.get('reasons', {})
        if abs(sum(reasons.values()) - 1.0) > 0.001:  # Allow small floating point error
            raise ValueError("legitimate_chargebacks.reasons probabilities must sum to 1")
    
    def _validate_pattern_injection_settings(self) -> None:
        """Validate pattern injection settings."""
        pattern = self.config['pattern_injection']
        
        # Validate customer ratio
        if not 0 <= pattern.get('customer_ratio', 0) <= 1:
            raise ValueError("pattern_injection.customer_ratio must be between 0 and 1")
        
        # Validate chargebacks in pattern range
        cb_in_pattern = pattern.get('chargebacks_in_pattern', {})
        if not (0 < cb_in_pattern.get('min', 0) <= cb_in_pattern.get('max', 0)):
            raise ValueError("Invalid chargebacks_in_pattern range")
        
        # Validate time window
        time_window = pattern.get('time_window', {})
        if not (0 < time_window.get('min', 0) <= time_window.get('max', 0)):
            raise ValueError("Invalid time_window range")
        
        # Validate repeat merchant probability
        if not 0 <= pattern.get('repeat_merchant_prob', 0) <= 1:
            raise ValueError("pattern_injection.repeat_merchant_prob must be between 0 and 1")
        
        # Validate chargeback delay
        delay = pattern.get('chargeback_delay', {})
        if not (0 < delay.get('min', 0) <= delay.get('max', 0)):
            raise ValueError("Invalid chargeback_delay range")
        
        # Validate pattern reasons probabilities sum to 1
        reasons = pattern.get('reasons', {})
        if abs(sum(reasons.values()) - 1.0) > 0.001:  # Allow small floating point error
            raise ValueError("pattern_injection.reasons probabilities must sum to 1")
    
    def _validate_probability_distributions(self) -> None:
        """Validate that all probability distributions sum to 1."""
        # This is a placeholder for additional probability distribution validations
        pass
    
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