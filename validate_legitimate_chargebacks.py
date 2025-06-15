import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
import numpy as np
from src.config.config_manager import ConfigurationManager
from src.engine.transaction_engine import TransactionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_modify_config():
    """Load the default config and modify it for legitimate chargeback testing."""
    # First create a temporary config file
    config_path = Path('test_config.yaml')
    
    # Load default config
    default_config_path = Path('src/config/default_config.yaml')
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set exact parameters as described in the validation text
    config['transactions']['num_transactions'] = 500000
    config['fraud_patterns']['total_fraud_ratio'] = 0.15  # 15% fraud
    
    # Set population size for large test
    config['population']['num_customers'] = 50000
    config['population']['num_merchants'] = 2500
    
    # Ensure legitimate chargeback rate is set to 2%
    config['transactions']['legitimate_chargebacks']['rate'] = 0.02
    config['transactions']['legitimate_chargebacks']['enabled'] = True
    
    # Write modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create test config
    test_config = ConfigurationManager(str(config_path))
    
    # Clean up temporary file
    config_path.unlink()
    
    return test_config

def count_legitimate_chargebacks(graph):
    """Count legitimate chargebacks and normal transactions in the graph."""
    normal_transactions = 0
    legitimate_chargebacks = 0
    total_transactions = 0
    fraudulent_transactions = 0
    
    for node, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'transaction':
            total_transactions += 1
            
            # Check if it's a chargeback transaction
            is_chargeback = attrs.get('is_chargeback', False)
            is_fraudulent = attrs.get('is_fraudulent', False)
            
            # Get customer information to determine if transaction is fraudulent
            if not is_fraudulent:
                customer_id = attrs.get('customer_id')
                if customer_id and customer_id in graph.nodes:
                    customer = graph.nodes[customer_id]
                    is_fraudulent = customer.get('is_fraudster', False) or customer.get('fraud_type') is not None
            
            if is_chargeback:
                if not is_fraudulent:
                    legitimate_chargebacks += 1
            else:
                if is_fraudulent:
                    fraudulent_transactions += 1
                else:
                    normal_transactions += 1
    
    return {
        'total_transactions': total_transactions,
        'normal_transactions': normal_transactions,
        'fraudulent_transactions': fraudulent_transactions,
        'legitimate_chargebacks': legitimate_chargebacks
    }

def run_single_test(config):
    """Run a single test and return the chargeback counts."""
    # Initialize and run transaction engine
    engine = TransactionEngine(config)
    
    # Generate base population
    engine.generate_base_population()
    
    # Generate normal transactions
    engine.generate_normal_transactions()
    
    # Inject fraud patterns
    engine._inject_patterns()
    
    # Get the graph and count transactions
    graph = engine.get_graph()
    counts = count_legitimate_chargebacks(graph)
    
    return counts

def create_chargeback_validation_plot(results, expected_chargebacks):
    """Create a plot showing legitimate chargeback validation across 10 runs."""
    plt.figure(figsize=(15, 10))
    
    # Extract data
    legitimate_chargebacks = [result['legitimate_chargebacks'] for result in results]
    normal_transactions = [result['normal_transactions'] for result in results]
    actual_rates = [cb/norm if norm > 0 else 0 for cb, norm in zip(legitimate_chargebacks, normal_transactions)]
    
    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top plot: Absolute numbers
    x_positions = range(1, 11)
    bars = ax1.bar(x_positions, legitimate_chargebacks, color='orange', alpha=0.7, label='Actual Legitimate Chargebacks')
    ax1.axhline(y=expected_chargebacks, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_chargebacks:,}')
    
    # Add value labels on bars
    for i, (pos, count) in enumerate(zip(x_positions, legitimate_chargebacks)):
        ax1.text(pos, count + 20, f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_title('Legitimate Chargeback Validation - Absolute Numbers', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Number of Legitimate Chargebacks')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'Run {i}' for i in x_positions])
    
    # Bottom plot: Percentage rates
    target_rate = 0.02  # 2%
    bars2 = ax2.bar(x_positions, [rate * 100 for rate in actual_rates], color='lightblue', alpha=0.7, label='Actual Rate (%)')
    ax2.axhline(y=target_rate * 100, color='red', linestyle='--', linewidth=2, label='Target: 2.0%')
    
    # Add percentage labels on bars
    for i, (pos, rate) in enumerate(zip(x_positions, actual_rates)):
        ax2.text(pos, rate * 100 + 0.01, f'{rate:.3%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_title('Legitimate Chargeback Rate Validation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Legitimate Chargeback Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'Run {i}' for i in x_positions])
    
    # Add summary statistics
    avg_chargebacks = np.mean(legitimate_chargebacks)
    std_chargebacks = np.std(legitimate_chargebacks)
    avg_rate = np.mean(actual_rates)
    std_rate = np.std(actual_rates)
    
    stats_text = f'''Summary Statistics:
Average Legitimate Chargebacks: {avg_chargebacks:.0f} (±{std_chargebacks:.0f})
Average Rate: {avg_rate:.4%} (±{std_rate:.4%})
Expected: {expected_chargebacks:,} chargebacks (2.00%)
Deviation: {avg_chargebacks - expected_chargebacks:+.0f} from expected'''
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    filename = 'legitimate_chargeback_validation_500k_transactions.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def main():
    logger.info("Starting legitimate chargeback validation...")
    logger.info("Configuration: 500,000 transactions, 15% fraud, 2% legitimate chargebacks")
    
    # Load and modify config
    config = load_and_modify_config()
    
    # Calculate expected values
    total_transactions = 500000
    fraud_ratio = 0.15
    chargeback_rate = 0.02
    
    fraudulent_transactions = int(total_transactions * fraud_ratio)
    normal_transactions = total_transactions - fraudulent_transactions
    expected_legitimate_chargebacks = int(normal_transactions * chargeback_rate)
    
    logger.info(f"Expected breakdown:")
    logger.info(f"  Total transactions: {total_transactions:,}")
    logger.info(f"  Fraudulent transactions: {fraudulent_transactions:,} ({fraud_ratio:.1%})")
    logger.info(f"  Normal transactions: {normal_transactions:,}")
    logger.info(f"  Expected legitimate chargebacks: {expected_legitimate_chargebacks:,} ({chargeback_rate:.1%} of normal)")
    
    # Run 10 validation tests
    results = []
    logger.info("\nRunning 10 validation tests...")
    
    for i in range(1, 11):
        logger.info(f"Running test {i}/10...")
        counts = run_single_test(config)
        results.append(counts)
        
        # Calculate actual rate
        actual_rate = counts['legitimate_chargebacks'] / counts['normal_transactions'] if counts['normal_transactions'] > 0 else 0
        deviation = counts['legitimate_chargebacks'] - expected_legitimate_chargebacks
        
        logger.info(f"  Test {i}: {counts['legitimate_chargebacks']:,} legitimate chargebacks ({actual_rate:.4%}) - deviation: {deviation:+d}")
    
    # Calculate and print summary statistics
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    legitimate_chargebacks = [result['legitimate_chargebacks'] for result in results]
    normal_transactions = [result['normal_transactions'] for result in results]
    actual_rates = [cb/norm for cb, norm in zip(legitimate_chargebacks, normal_transactions)]
    
    avg_chargebacks = np.mean(legitimate_chargebacks)
    std_chargebacks = np.std(legitimate_chargebacks)
    avg_rate = np.mean(actual_rates)
    std_rate = np.std(actual_rates)
    min_chargebacks = np.min(legitimate_chargebacks)
    max_chargebacks = np.max(legitimate_chargebacks)
    
    logger.info(f"Expected legitimate chargebacks: {expected_legitimate_chargebacks:,}")
    logger.info(f"Actual legitimate chargebacks:")
    logger.info(f"  Average: {avg_chargebacks:.1f} (±{std_chargebacks:.1f})")
    logger.info(f"  Range: {min_chargebacks:,} - {max_chargebacks:,}")
    logger.info(f"  Average deviation: {avg_chargebacks - expected_legitimate_chargebacks:+.1f}")
    logger.info(f"")
    logger.info(f"Expected rate: 2.000%")
    logger.info(f"Actual rate:")
    logger.info(f"  Average: {avg_rate:.4%} (±{std_rate:.4%})")
    logger.info(f"  Range: {min(actual_rates):.4%} - {max(actual_rates):.4%}")
    
    # Statistical significance test
    deviation_ratio = abs(avg_chargebacks - expected_legitimate_chargebacks) / expected_legitimate_chargebacks
    logger.info(f"")
    logger.info(f"Relative deviation: {deviation_ratio:.4%}")
    if deviation_ratio < 0.01:  # Less than 1% deviation
        logger.info("✓ VALIDATION PASSED: Deviation is within acceptable range (<1%)")
    elif deviation_ratio < 0.05:  # Less than 5% deviation
        logger.info("⚠ VALIDATION WARNING: Deviation is moderate (1-5%)")
    else:
        logger.info("✗ VALIDATION FAILED: Deviation is too large (>5%)")
    
    # Create validation plot
    logger.info("\nCreating validation plot...")
    filename = create_chargeback_validation_plot(results, expected_legitimate_chargebacks)
    logger.info(f"Plot saved as '{filename}'")
    
    logger.info("\nValidation complete!")

if __name__ == "__main__":
    main() 