import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
import numpy as np
from src.config.config_manager import ConfigurationManager
from src.engine.transaction_engine import TransactionEngine
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_modify_config(num_transactions=100, fraud_ratio=0.2):
    """Load the default config and modify it for testing."""
    # First create a temporary config file
    config_path = Path('test_config.yaml')
    
    # Load default config
    default_config_path = Path('src/config/default_config.yaml')
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for specified test parameters
    config['transactions']['num_transactions'] = num_transactions
    config['fraud_patterns']['total_fraud_ratio'] = fraud_ratio
    
    # Scale population size based on transaction count
    # Use reasonable ratios: roughly 10 transactions per customer, 20 customers per merchant
    num_customers = max(10, num_transactions // 10)
    num_merchants = max(5, num_customers // 20)
    
    config['population']['num_customers'] = num_customers
    config['population']['num_merchants'] = num_merchants
    
    # Adjust pattern settings to better control transaction counts
    # Serial Chargeback (20% of fraud transactions)
    config['fraud_patterns']['serial_chargeback']['transactions_in_pattern']['min'] = 2
    config['fraud_patterns']['serial_chargeback']['transactions_in_pattern']['max'] = 2
    
    # BIN Attack (20% of fraud transactions)
    config['fraud_patterns']['bin_attack']['num_cards']['min'] = 2
    config['fraud_patterns']['bin_attack']['num_cards']['max'] = 2
    
    # Friendly Fraud (60% of fraud transactions)
    config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['min'] = 3
    config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['max'] = 3
    config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['min'] = 1
    config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['max'] = 1
    
    # Adjust chargeback probabilities to prevent excess transactions
    config['fraud_patterns']['friendly_fraud']['chargeback_probability'] = 1.0
    config['fraud_patterns']['bin_attack']['chargeback_rate'] = 0.0  # No chargebacks for BIN attack in test
    
    logger.info(f"Configured for {num_transactions} transactions with {fraud_ratio:.1%} fraud ratio")
    logger.info(f"Population: {num_customers} customers, {num_merchants} merchants")
    
    # Write modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create test config
    test_config = ConfigurationManager(str(config_path))
    
    # Clean up temporary file
    config_path.unlink()
    
    return test_config

def count_transactions(graph):
    """Count legitimate and fraudulent transactions in the graph."""
    legitimate_count = 0
    fraud_count = 0
    
    for node, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'transaction' and not attrs.get('is_chargeback', False):
            # A transaction is fraudulent if:
            # 1. It's marked as fraudulent directly
            # 2. OR it belongs to a fraudulent customer
            is_fraudulent = attrs.get('is_fraudulent', False)
            
            if not is_fraudulent:
                # Check if the customer is fraudulent
                customer_id = attrs.get('customer_id')
                if customer_id:
                    customer = graph.nodes[customer_id]
                    is_fraudulent = customer.get('is_fraudster', False) or customer.get('fraud_type') is not None
            
            if is_fraudulent:
                fraud_count += 1
            else:
                legitimate_count += 1
    
    return legitimate_count, fraud_count

def run_single_test(config):
    """Run a single test and return the transaction counts."""
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
    legitimate_count, fraud_count = count_transactions(graph)
    
    return legitimate_count, fraud_count

def create_stacked_validation_plot(results, target_fraud, target_ratio, num_transactions):
    """Create a stacked bar plot showing 10 runs with legitimate (bottom) and fraud (top) transactions."""
    plt.figure(figsize=(14, 8))
    
    # Extract data
    legitimate_counts = [result[0] for result in results]
    fraud_counts = [result[1] for result in results]
    totals = [legit + fraud for legit, fraud in zip(legitimate_counts, fraud_counts)]
    fraud_ratios = [fraud/total if total > 0 else 0 for fraud, total in zip(fraud_counts, totals)]
    
    # Create x positions for bars
    x_positions = range(1, 11)  # Run 1 through Run 10
    
    # Create stacked bar chart
    # Bottom bars (legitimate) in light blue
    bars_legit = plt.bar(x_positions, legitimate_counts, 
                        color='lightblue', label='Legitimate Transactions', alpha=0.8)
    
    # Top bars (fraud) in red, stacked on top of legitimate
    bars_fraud = plt.bar(x_positions, fraud_counts, 
                        bottom=legitimate_counts, color='red', 
                        label='Fraud Transactions', alpha=0.8)
    
    # Add percentage labels on top of each bar
    # Adjust label positioning and formatting based on transaction count
    max_total = max(totals) if totals else 100
    label_offset = max(max_total * 0.03, 2)  # 3% of max value, minimum 2
    
    # Adjust font size based on number of runs and transaction count
    if num_transactions <= 50:
        fontsize = 8
    elif num_transactions <= 500:
        fontsize = 9
    else:
        fontsize = 10
    
    for i, (fraud_ratio, total) in enumerate(zip(fraud_ratios, totals)):
        # Use more compact formatting for small numbers
        if total < 1000:
            label_text = f'{fraud_ratio:.1%}\n({total} total)'
        else:
            label_text = f'{fraud_ratio:.1%}\n({total:,} total)'
            
        plt.text(x_positions[i], total + label_offset, label_text, 
                ha='center', va='bottom', fontweight='bold', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Customize plot
    plt.title(f'Fraud Ratio Validation - 10 Runs\nTarget: {num_transactions:,} transactions, {target_ratio:.1%} fraud ratio', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.legend(loc='lower right')
    
    # Set x-axis labels
    plt.xticks(x_positions, [f'Run {i}' for i in x_positions])
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis appropriately based on transaction count
    if num_transactions >= 10000:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    else:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    # Add summary statistics as text in bottom left
    avg_fraud_ratio = np.mean(fraud_ratios)
    std_fraud_ratio = np.std(fraud_ratios)
    min_fraud_ratio = np.min(fraud_ratios)
    max_fraud_ratio = np.max(fraud_ratios)
    
    stats_text = f'''Summary Statistics:
Average Fraud Ratio: {avg_fraud_ratio:.2%}
Std Deviation: {std_fraud_ratio:.3%}
Range: {min_fraud_ratio:.2%} - {max_fraud_ratio:.2%}'''
    
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Adjust layout and save with descriptive filename
    plt.tight_layout()
    filename = f'fraud_ratio_validation_{num_transactions}_transactions_{target_ratio:.0%}_fraud.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate fraud ratio generation')
    parser.add_argument('--transactions', type=int, default=100, 
                       help='Number of transactions to generate (default: 100)')
    parser.add_argument('--fraud-ratio', type=float, default=0.2,
                       help='Target fraud ratio (default: 0.2)')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of validation runs (default: 10)')
    
    args = parser.parse_args()
    
    # Load and modify config
    logger.info("Loading and modifying configuration...")
    config = load_and_modify_config(args.transactions, args.fraud_ratio)
    
    # Calculate target values
    target_transactions = args.transactions
    target_ratio = args.fraud_ratio
    target_fraud = int(target_transactions * target_ratio)
    
    logger.info(f"Target total transactions: {target_transactions:,}")
    logger.info(f"Target fraud ratio: {target_ratio:.1%}")
    logger.info(f"Target fraud transactions: {target_fraud:,}")
    
    # Run validation tests
    results = []
    logger.info(f"\nRunning {args.runs} validation tests...")
    
    for i in range(1, args.runs + 1):
        logger.info(f"Running test {i}/{args.runs}...")
        legitimate_count, fraud_count = run_single_test(config)
        total_count = legitimate_count + fraud_count
        fraud_ratio = fraud_count / total_count if total_count > 0 else 0
        
        results.append((legitimate_count, fraud_count))
        logger.info(f"  Test {i}: {total_count:,} total, {fraud_count:,} fraud ({fraud_ratio:.2%})")
    
    # Print summary
    logger.info("\nSummary of all runs:")
    fraud_ratios = [fraud/(legit+fraud) if (legit+fraud) > 0 else 0 for legit, fraud in results]
    avg_ratio = np.mean(fraud_ratios)
    std_ratio = np.std(fraud_ratios)
    
    logger.info(f"Average fraud ratio: {avg_ratio:.2%}")
    logger.info(f"Standard deviation: {std_ratio:.3%}")
    logger.info(f"Range: {min(fraud_ratios):.2%} - {max(fraud_ratios):.2%}")
    logger.info(f"Target ratio: {target_ratio:.1%}")
    
    # Create stacked validation plot
    logger.info("\nCreating stacked validation plot...")
    filename = create_stacked_validation_plot(results, target_fraud, target_ratio, target_transactions)
    logger.info(f"Plot saved as '{filename}'")

if __name__ == "__main__":
    main() 