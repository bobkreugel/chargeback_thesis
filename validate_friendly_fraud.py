import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def load_config():
    """Load the configuration file."""
    # First try to load the config that was used to generate the dataset
    dataset_config_path = Path('output/dataset/config.json')
    if dataset_config_path.exists():
        import json
        print("Loading configuration from dataset (the config used to generate the data)...")
        with open(dataset_config_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback to current config if dataset config not found
        print("Dataset config not found, loading current configuration...")
        config_path = Path('src/config/default_config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

def load_graph(filepath):
    """Load the NetworkX graph from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_friendly_fraud_patterns(graph, config):
    """
    Analyze friendly fraud patterns according to the configuration:
    Uses dynamic values from config file for all validation criteria.
    """
    
    # Get configuration values for dynamic validation
    expected_ff_ratio = config['fraud_patterns']['pattern_distribution']['friendly_fraud']
    legit_period_min = config['fraud_patterns']['friendly_fraud']['legitimate_period']['min']
    legit_period_max = config['fraud_patterns']['friendly_fraud']['legitimate_period']['max']
    legit_tx_min = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['min']
    legit_tx_max = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['max']
    fraud_tx_min = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['min']
    fraud_tx_max = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['max']
    chargeback_prob = config['fraud_patterns']['friendly_fraud']['chargeback_probability']
    cb_delay_min = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['min']
    cb_delay_max = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['max']
    merchant_reuse_prob = config['fraud_patterns']['friendly_fraud']['merchant_reuse_prob']
    
    print(f"Validating against config expectations:")
    print(f"  - Volume: {expected_ff_ratio*100:.1f}% of all fraud transactions")
    print(f"  - Legitimate period: {legit_period_min}-{legit_period_max} days")
    print(f"  - Legitimate transactions: {legit_tx_min}-{legit_tx_max} per customer")
    print(f"  - Fraudulent transactions: {fraud_tx_min}-{fraud_tx_max} per customer")
    print(f"  - Chargeback ratio: {chargeback_prob*100:.1f}%")
    print(f"  - Chargeback delay: {cb_delay_min}-{cb_delay_max} days")
    print(f"  - Merchant reuse: {merchant_reuse_prob*100:.1f}%")
    
    # Get friendly fraud customers
    friendly_fraud_customers = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'customer' and attrs.get('fraud_type') == 'friendly_fraud':
            friendly_fraud_customers.append(node)
    
    print(f"\nFound {len(friendly_fraud_customers)} friendly fraud customers")
    
    # Initialize analysis data
    customer_analysis = {}
    all_transactions = []
    all_fraud_transactions = []
    friendly_fraud_transactions = []
    
    # Use the same counting method as generate_data.py
    # Count all fraudulent transactions by looking at customer fraud_type (not individual transaction attributes)
    for node, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'transaction':
            # Skip chargeback transactions for fraud pattern counting (same as generate_data.py)
            if attrs.get('is_chargeback', False):
                continue
                
            # Get customer who made the transaction
            card_edges = [e for e in graph.in_edges(node, data=True) if e[2].get('edge_type') == 'card_to_transaction']
            if card_edges:
                card_id = card_edges[0][0]
                customer_edges = [e for e in graph.in_edges(card_id, data=True) if e[2].get('edge_type') == 'customer_to_card']
                if customer_edges:
                    customer_id = customer_edges[0][0]
                    customer_attr = graph.nodes[customer_id]
                    
                    if customer_attr.get('is_fraudster', False):
                        # This is a fraudulent transaction (based on customer fraud_type)
                        fraud_type = customer_attr.get('fraud_type')
                        if fraud_type == 'friendly_fraud':
                            friendly_fraud_transactions.append(attrs)
                        
                        # Add to all fraud transactions
                        all_fraud_transactions.append(attrs)
    
    # Analyze each friendly fraud customer
    for customer_id in friendly_fraud_customers:
        customer_data = {
            'customer_id': customer_id,
            'legitimate_transactions': [],
            'fraudulent_transactions': [],
            'chargebacks': [],
            'merchants_used': set(),
            'legitimate_period_days': 0,
            'fraudulent_period_days': 0
        }
        
        # Get customer's card
        card_id = None
        for edge in graph.out_edges(customer_id):
            if graph.nodes[edge[1]].get('node_type') == 'card':
                card_id = edge[1]
                break
        
        if not card_id:
            continue
            
        # Get all transactions for this card
        transactions = []
        for edge in graph.out_edges(card_id):
            if graph.nodes[edge[1]].get('node_type') == 'transaction':
                tx_node = edge[1]
                tx_attrs = graph.nodes[tx_node].copy()
                tx_attrs['transaction_id'] = tx_node
                
                # Find merchant for this transaction
                for merchant_edge in graph.in_edges(tx_node):
                    if graph.nodes[merchant_edge[0]].get('node_type') == 'merchant':
                        tx_attrs['merchant_id'] = merchant_edge[0]
                        break
                
                transactions.append(tx_attrs)
        
        # Sort transactions by timestamp
        transactions.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        # Separate legitimate and fraudulent transactions
        for tx in transactions:
            if tx.get('is_chargeback', False):
                customer_data['chargebacks'].append(tx)
            elif tx.get('is_fraudulent', False):
                customer_data['fraudulent_transactions'].append(tx)
            else:
                customer_data['legitimate_transactions'].append(tx)
            
            # Track merchants used
            if 'merchant_id' in tx:
                customer_data['merchants_used'].add(tx['merchant_id'])
            
            all_transactions.append(tx)
        
        # Calculate time periods - improved calculation
        if customer_data['legitimate_transactions']:
            legit_times = [tx['timestamp'] for tx in customer_data['legitimate_transactions']]
            if len(legit_times) > 1:
                customer_data['legitimate_period_days'] = (max(legit_times) - min(legit_times)).days
            else:
                # Single transaction, period is 0
                customer_data['legitimate_period_days'] = 0
                
        if customer_data['fraudulent_transactions']:
            fraud_times = [tx['timestamp'] for tx in customer_data['fraudulent_transactions']]
            if len(fraud_times) > 1:
                customer_data['fraudulent_period_days'] = (max(fraud_times) - min(fraud_times)).days
            else:
                # Single transaction, period is 0
                customer_data['fraudulent_period_days'] = 0
        
        customer_analysis[customer_id] = customer_data
    
    return customer_analysis, all_transactions, all_fraud_transactions, friendly_fraud_transactions

def calculate_statistics(customer_analysis, all_fraud_transactions, friendly_fraud_transactions, config):
    """Calculate validation statistics."""
    stats = {
        'volume': {},
        'phasing': {},
        'chargebacks': {},
        'merchant_reuse': {}
    }
    
    # 1. Volume Analysis
    total_fraud_txs = len(all_fraud_transactions)
    friendly_fraud_txs = len(friendly_fraud_transactions)
    expected_friendly_fraud_ratio = config['fraud_patterns']['pattern_distribution']['friendly_fraud']
    actual_friendly_fraud_ratio = friendly_fraud_txs / total_fraud_txs if total_fraud_txs > 0 else 0
    
    stats['volume'] = {
        'total_fraud_transactions': total_fraud_txs,
        'friendly_fraud_transactions': friendly_fraud_txs,
        'expected_ratio': expected_friendly_fraud_ratio,
        'actual_ratio': actual_friendly_fraud_ratio,
        'ratio_difference': abs(actual_friendly_fraud_ratio - expected_friendly_fraud_ratio)
    }
    
    # 2. Phasing Analysis
    legitimate_counts = []
    fraudulent_counts = []
    legitimate_periods = []
    fraudulent_periods = []
    
    config_legit_min = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['min']
    config_legit_max = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['max']
    config_fraud_min = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['min']
    config_fraud_max = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['max']
    config_legit_period_min = config['fraud_patterns']['friendly_fraud']['legitimate_period']['min']
    config_legit_period_max = config['fraud_patterns']['friendly_fraud']['legitimate_period']['max']
    
    for customer_data in customer_analysis.values():
        legit_count = len(customer_data['legitimate_transactions'])
        fraud_count = len(customer_data['fraudulent_transactions'])
        
        legitimate_counts.append(legit_count)
        fraudulent_counts.append(fraud_count)
        legitimate_periods.append(customer_data['legitimate_period_days'])
        fraudulent_periods.append(customer_data['fraudulent_period_days'])
    
    stats['phasing'] = {
        'customers_analyzed': len(customer_analysis),
        'legitimate_transactions': {
            'avg': np.mean(legitimate_counts) if legitimate_counts else 0,
            'min': min(legitimate_counts) if legitimate_counts else 0,
            'max': max(legitimate_counts) if legitimate_counts else 0,
            'within_config': sum(1 for c in legitimate_counts if config_legit_min <= c <= config_legit_max),
            'config_min': config_legit_min,
            'config_max': config_legit_max
        },
        'fraudulent_transactions': {
            'avg': np.mean(fraudulent_counts) if fraudulent_counts else 0,
            'min': min(fraudulent_counts) if fraudulent_counts else 0,
            'max': max(fraudulent_counts) if fraudulent_counts else 0,
            'within_config': sum(1 for c in fraudulent_counts if config_fraud_min <= c <= config_fraud_max),
            'config_min': config_fraud_min,
            'config_max': config_fraud_max
        },
        'legitimate_periods': {
            'avg': np.mean(legitimate_periods) if legitimate_periods else 0,
            'min': min(legitimate_periods) if legitimate_periods else 0,
            'max': max(legitimate_periods) if legitimate_periods else 0,
            'within_config': sum(1 for p in legitimate_periods if config_legit_period_min <= p <= config_legit_period_max),
            'config_min': config_legit_period_min,
            'config_max': config_legit_period_max
        }
    }
    
    # 3. Chargeback Analysis
    total_fraudulent_txs = sum(len(customer_data['fraudulent_transactions']) for customer_data in customer_analysis.values())
    total_chargebacks = sum(len(customer_data['chargebacks']) for customer_data in customer_analysis.values())
    
    expected_chargeback_prob = config['fraud_patterns']['friendly_fraud']['chargeback_probability']
    actual_chargeback_ratio = total_chargebacks / total_fraudulent_txs if total_fraudulent_txs > 0 else 0
    
    # Calculate chargeback delays
    chargeback_delays = []
    config_delay_min = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['min']
    config_delay_max = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['max']
    
    for customer_data in customer_analysis.values():
        for chargeback in customer_data['chargebacks']:
            if 'original_transaction' in chargeback:
                # Find the original transaction
                original_tx = None
                for tx in customer_data['fraudulent_transactions']:
                    if tx['transaction_id'] == chargeback['original_transaction']:
                        original_tx = tx
                        break
                
                if original_tx:
                    delay_days = (chargeback['timestamp'] - original_tx['timestamp']).days
                    chargeback_delays.append(delay_days)
    
    stats['chargebacks'] = {
        'total_fraudulent_transactions': total_fraudulent_txs,
        'total_chargebacks': total_chargebacks,
        'expected_probability': expected_chargeback_prob,
        'actual_ratio': actual_chargeback_ratio,
        'ratio_difference': abs(actual_chargeback_ratio - expected_chargeback_prob),
        'delays': {
            'avg': np.mean(chargeback_delays) if chargeback_delays else 0,
            'min': min(chargeback_delays) if chargeback_delays else 0,
            'max': max(chargeback_delays) if chargeback_delays else 0,
            'within_config': sum(1 for d in chargeback_delays if config_delay_min <= d <= config_delay_max),
            'total_delays': len(chargeback_delays),
            'config_min': config_delay_min,
            'config_max': config_delay_max
        }
    }
    
    # 4. Merchant Reuse Analysis
    total_transactions_per_customer = []
    unique_merchants_per_customer = []
    reuse_rates = []
    
    for customer_data in customer_analysis.values():
        total_txs = len(customer_data['legitimate_transactions']) + len(customer_data['fraudulent_transactions'])
        unique_merchants = len(customer_data['merchants_used'])
        
        if total_txs > 0:
            total_transactions_per_customer.append(total_txs)
            unique_merchants_per_customer.append(unique_merchants)
            reuse_rate = (total_txs - unique_merchants) / total_txs if total_txs > 0 else 0
            reuse_rates.append(reuse_rate)
    
    expected_reuse_prob = config['fraud_patterns']['friendly_fraud']['merchant_reuse_prob']
    actual_avg_reuse = np.mean(reuse_rates) if reuse_rates else 0
    
    stats['merchant_reuse'] = {
        'expected_probability': expected_reuse_prob,
        'actual_average': actual_avg_reuse,
        'difference': abs(actual_avg_reuse - expected_reuse_prob),
        'customer_rates': reuse_rates,
        'avg_transactions_per_customer': np.mean(total_transactions_per_customer) if total_transactions_per_customer else 0,
        'avg_unique_merchants_per_customer': np.mean(unique_merchants_per_customer) if unique_merchants_per_customer else 0
    }
    
    return stats

def create_visualizations(stats, config):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(15, 8))
    
    # 1. Volume Comparison
    ax1 = plt.subplot(2, 3, 1)
    labels = ['Expected', 'Actual']
    values = [stats['volume']['expected_ratio'] * 100, stats['volume']['actual_ratio'] * 100]
    bars = ax1.bar(labels, values, color=['lightblue', 'salmon'])
    ax1.set_title('Friendly Fraud Volume\n(% of all fraud transactions)')
    ax1.set_ylabel('Percentage')
    ax1.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Chargeback Analysis
    ax2 = plt.subplot(2, 3, 2)
    cb_labels = ['Expected', 'Actual']
    cb_values = [stats['chargebacks']['expected_probability'] * 100, 
                 stats['chargebacks']['actual_ratio'] * 100]
    bars = ax2.bar(cb_labels, cb_values, color=['lightblue', 'orange'])
    ax2.set_title('Chargeback Ratio')
    ax2.set_ylabel('Percentage')
    ax2.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. Chargeback Delays
    ax3 = plt.subplot(2, 3, 3)
    if stats['chargebacks']['delays']['total_delays'] > 0:
        delay_avg = stats['chargebacks']['delays']['avg']
        delay_min = stats['chargebacks']['delays']['config_min']
        delay_max = stats['chargebacks']['delays']['config_max']
        
        ax3.bar(['Actual Avg'], [delay_avg], color='gold')
        ax3.axhline(y=delay_min, color='r', linestyle='--', label=f'Config Min ({delay_min}d)')
        ax3.axhline(y=delay_max, color='r', linestyle='--', label=f'Config Max ({delay_max}d)')
        ax3.text(0, delay_avg, f'{delay_avg:.1f}d', ha='center', va='bottom')
    ax3.set_title('Chargeback Delays')
    ax3.set_ylabel('Days')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Merchant Reuse
    ax4 = plt.subplot(2, 3, 4)
    reuse_labels = ['Expected', 'Actual']
    reuse_values = [stats['merchant_reuse']['expected_probability'] * 100,
                    stats['merchant_reuse']['actual_average'] * 100]
    bars = ax4.bar(reuse_labels, reuse_values, color=['lightblue', 'lightgreen'])
    ax4.set_title('Merchant Reuse Rate')
    ax4.set_ylabel('Percentage')
    ax4.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Legitimate Period Distribution
    ax5 = plt.subplot(2, 3, 5)
    legit_period_avg = stats['phasing']['legitimate_periods']['avg']
    legit_period_min = stats['phasing']['legitimate_periods']['config_min']
    legit_period_max = stats['phasing']['legitimate_periods']['config_max']
    
    ax5.bar(['Actual Avg'], [legit_period_avg], color='lightcyan')
    ax5.axhline(y=legit_period_min, color='r', linestyle='--', label=f'Config Min ({legit_period_min}d)')
    ax5.axhline(y=legit_period_max, color='r', linestyle='--', label=f'Config Max ({legit_period_max}d)')
    ax5.set_title('Legitimate Period Length')
    ax5.set_ylabel('Days')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.text(0, legit_period_avg, f'{legit_period_avg:.1f}d', ha='center', va='bottom')
    
    plt.suptitle('Friendly Fraud Pattern Validation', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('friendly_fraud_validation.png', bbox_inches='tight', dpi=300)
    plt.close()

def print_detailed_report(stats):
    """Print detailed validation report."""
    print("\n" + "="*80)
    print("FRIENDLY FRAUD PATTERN VALIDATION REPORT")
    print("="*80)
    
    # Volume Analysis
    print(f"\n1. VOLUME ANALYSIS:")
    print(f"   Total fraud transactions: {stats['volume']['total_fraud_transactions']}")
    print(f"   Friendly fraud transactions: {stats['volume']['friendly_fraud_transactions']}")
    print(f"   Expected ratio: {stats['volume']['expected_ratio']*100:.1f}%")
    print(f"   Actual ratio: {stats['volume']['actual_ratio']*100:.1f}%")
    print(f"   Difference: {stats['volume']['ratio_difference']*100:.1f}%")
    print(f"   ✓ PASSED" if stats['volume']['ratio_difference'] < 0.05 else f"   ✗ FAILED")
    
    # Phasing Analysis
    print(f"\n2. PHASING ANALYSIS:")
    print(f"   Customers analyzed: {stats['phasing']['customers_analyzed']}")
    
    print(f"\n   Legitimate transactions per customer:")
    legit = stats['phasing']['legitimate_transactions']
    print(f"     Average: {legit['avg']:.1f}")
    print(f"     Range: {legit['min']}-{legit['max']}")
    print(f"     Config range: {legit['config_min']}-{legit['config_max']}")
    print(f"     Within config: {legit['within_config']}/{stats['phasing']['customers_analyzed']}")
    print(f"     ✓ PASSED" if legit['within_config'] >= stats['phasing']['customers_analyzed'] * 0.8 else f"     ✗ FAILED")
    
    print(f"\n   Fraudulent transactions per customer:")
    fraud = stats['phasing']['fraudulent_transactions']
    print(f"     Average: {fraud['avg']:.1f}")
    print(f"     Range: {fraud['min']}-{fraud['max']}")
    print(f"     Config range: {fraud['config_min']}-{fraud['config_max']}")
    print(f"     Within config: {fraud['within_config']}/{stats['phasing']['customers_analyzed']}")
    print(f"     ✓ PASSED" if fraud['within_config'] >= stats['phasing']['customers_analyzed'] * 0.8 else f"     ✗ FAILED")
    
    print(f"\n   Legitimate period length:")
    period = stats['phasing']['legitimate_periods']
    print(f"     Average: {period['avg']:.1f} days")
    print(f"     Range: {period['min']}-{period['max']} days")
    print(f"     Config range: {period['config_min']}-{period['config_max']} days")
    print(f"     Within config: {period['within_config']}/{stats['phasing']['customers_analyzed']}")
    print(f"     ✓ PASSED" if period['within_config'] >= stats['phasing']['customers_analyzed'] * 0.8 else f"     ✗ FAILED")
    
    # Chargeback Analysis
    print(f"\n3. CHARGEBACK ANALYSIS:")
    cb = stats['chargebacks']
    print(f"   Total fraudulent transactions: {cb['total_fraudulent_transactions']}")
    print(f"   Total chargebacks: {cb['total_chargebacks']}")
    print(f"   Expected probability: {cb['expected_probability']*100:.1f}%")
    print(f"   Actual ratio: {cb['actual_ratio']*100:.1f}%")
    print(f"   Difference: {cb['ratio_difference']*100:.1f}%")
    print(f"   ✓ PASSED" if cb['ratio_difference'] < 0.1 else f"   ✗ FAILED")
    
    print(f"\n   Chargeback delays:")
    delays = cb['delays']
    print(f"     Average: {delays['avg']:.1f} days")
    print(f"     Range: {delays['min']}-{delays['max']} days")
    print(f"     Config range: {delays['config_min']}-{delays['config_max']} days")
    print(f"     Within config: {delays['within_config']}/{delays['total_delays']}")
    if delays['total_delays'] > 0:
        print(f"     ✓ PASSED" if delays['within_config'] >= delays['total_delays'] * 0.8 else f"     ✗ FAILED")
    
    # Merchant Reuse Analysis
    print(f"\n4. MERCHANT REUSE ANALYSIS:")
    reuse = stats['merchant_reuse']
    print(f"   Expected probability: {reuse['expected_probability']*100:.1f}%")
    print(f"   Actual average: {reuse['actual_average']*100:.1f}%")
    print(f"   Difference: {reuse['difference']*100:.1f}%")
    print(f"   Avg transactions per customer: {reuse['avg_transactions_per_customer']:.1f}")
    print(f"   Avg unique merchants per customer: {reuse['avg_unique_merchants_per_customer']:.1f}")
    print(f"   ✓ PASSED" if reuse['difference'] < 0.05 else f"   ✗ FAILED")

def main():
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Path to the generated graph pickle file
    graph_path = Path('output/dataset/transaction_graph.gpickle')
    
    if not graph_path.exists():
        print(f"Error: Could not find graph file at {graph_path}")
        return
    
    # Load the graph
    print("Loading transaction graph...")
    graph = load_graph(graph_path)
    
    # Analyze friendly fraud patterns
    print("Analyzing friendly fraud patterns...")
    customer_analysis, all_transactions, all_fraud_transactions, friendly_fraud_transactions = analyze_friendly_fraud_patterns(graph, config)
    
    if not customer_analysis:
        print("No friendly fraud customers found in the dataset!")
        return
    
    # Calculate statistics
    print("Calculating validation statistics...")
    stats = calculate_statistics(customer_analysis, all_fraud_transactions, friendly_fraud_transactions, config)
    
    # Print detailed report
    print_detailed_report(stats)
    
    # Create visualizations
    print("\nCreating validation visualizations...")
    create_visualizations(stats, config)
    print("Validation plots saved as 'friendly_fraud_validation.png'")

if __name__ == '__main__':
    main()
