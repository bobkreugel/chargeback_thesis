#!/usr/bin/env python3
"""
Friendly Fraud Pattern Validation Script
Runs multiple experiments with large datasets to validate configuration compliance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import our modules
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_manager import ConfigurationManager
from src.engine.transaction_engine import TransactionEngine

def run_experiment(num_transactions, run_id):
    """Run a single experiment and return validation metrics."""
    print(f"Running experiment {run_id} with {num_transactions:,} transactions...")
    
    # Create temporary config for this run
    config_manager = ConfigurationManager()
    
    # Modify config for this experiment
    config = config_manager.get_config()
    config['transactions']['num_transactions'] = num_transactions
    
    # Initialize engine
    engine = TransactionEngine(config_manager)
    
    # Generate data
    engine.generate_base_population()
    engine.generate_normal_transactions()
    engine._inject_patterns()
    
    # Get the graph
    graph = engine.get_graph()
    
    # Analyze friendly fraud patterns
    patterns = []
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'customer' and 
            attr.get('fraud_type') == 'friendly_fraud'):
            
            customer_id = node
            customer_data = attr
            
            # Find related data
            cards = []
            for edge in graph.out_edges(customer_id, data=True):
                if edge[2].get('edge_type') == 'customer_to_card':
                    cards.append(edge[1])
            
            transactions = []
            chargebacks = []
            merchants_used = set()
            
            for card_id in cards:
                for edge in graph.out_edges(card_id, data=True):
                    if edge[2].get('edge_type') == 'card_to_transaction':
                        tx_id = edge[1]
                        tx_data = graph.nodes[tx_id]
                        
                        # Find merchant for this transaction
                        merchant_id = None
                        for merchant_edge in graph.in_edges(tx_id, data=True):
                            if graph.nodes[merchant_edge[0]].get('node_type') == 'merchant':
                                merchant_id = merchant_edge[0]
                                merchants_used.add(merchant_id)
                                break
                        
                        tx_data_copy = tx_data.copy()
                        tx_data_copy['transaction_id'] = tx_id
                        tx_data_copy['merchant_id'] = merchant_id
                        
                        if tx_data.get('is_chargeback', False):
                            chargebacks.append(tx_data_copy)
                        else:
                            transactions.append(tx_data_copy)
            
            if transactions:
                # Separate legitimate and fraudulent transactions
                legitimate_txs = [tx for tx in transactions if not tx.get('is_fraudulent', False)]
                fraudulent_txs = [tx for tx in transactions if tx.get('is_fraudulent', False)]
                
                patterns.append({
                    'customer_id': customer_id,
                    'legitimate_transactions': legitimate_txs,
                    'fraudulent_transactions': fraudulent_txs,
                    'chargebacks': chargebacks,
                    'merchants_used': merchants_used
                })
    
    # Calculate validation metrics
    metrics = {
        'run_id': run_id,
        'num_transactions': num_transactions,
        'num_patterns': len(patterns),
        'legitimate_tx_counts': [],
        'fraudulent_tx_counts': [],
        'legitimate_periods': [],
        'fraudulent_periods': [],
        'chargeback_ratios': [],
        'chargeback_delays': [],
        'merchant_reuse_rates': [],
        'transaction_amounts': []
    }
    
    for pattern in patterns:
        # Transaction counts
        legit_count = len(pattern['legitimate_transactions'])
        fraud_count = len(pattern['fraudulent_transactions'])
        metrics['legitimate_tx_counts'].append(legit_count)
        metrics['fraudulent_tx_counts'].append(fraud_count)
        
        # Time periods
        if pattern['legitimate_transactions']:
            legit_times = [tx['timestamp'] for tx in pattern['legitimate_transactions'] if tx.get('timestamp')]
            if len(legit_times) > 1:
                legit_period = (max(legit_times) - min(legit_times)).days
                metrics['legitimate_periods'].append(legit_period)
            else:
                metrics['legitimate_periods'].append(0)
        
        if pattern['fraudulent_transactions']:
            fraud_times = [tx['timestamp'] for tx in pattern['fraudulent_transactions'] if tx.get('timestamp')]
            if len(fraud_times) > 1:
                fraud_period = (max(fraud_times) - min(fraud_times)).days
                metrics['fraudulent_periods'].append(fraud_period)
            else:
                metrics['fraudulent_periods'].append(0)
        
        # Chargeback analysis
        total_fraud_txs = len(pattern['fraudulent_transactions'])
        total_chargebacks = len(pattern['chargebacks'])
        if total_fraud_txs > 0:
            chargeback_ratio = total_chargebacks / total_fraud_txs
            metrics['chargeback_ratios'].append(chargeback_ratio)
        
        # Chargeback delays
        for chargeback in pattern['chargebacks']:
            original_tx_id = chargeback.get('original_transaction')
            if original_tx_id:
                for tx in pattern['fraudulent_transactions']:
                    if tx['transaction_id'] == original_tx_id:
                        if chargeback.get('timestamp') and tx.get('timestamp'):
                            delay = (chargeback['timestamp'] - tx['timestamp']).days
                            metrics['chargeback_delays'].append(delay)
                        break
        
        # Merchant reuse analysis - calculate actual reuse rate
        all_transactions = pattern['legitimate_transactions'] + pattern['fraudulent_transactions']
        if len(all_transactions) > 1:
            # Sort transactions by timestamp to get correct sequence
            sorted_txs = sorted(all_transactions, key=lambda x: x['timestamp'])
            merchant_sequence = [tx['merchant_id'] for tx in sorted_txs]
            reuse_count = 0
            for i in range(1, len(merchant_sequence)):
                if merchant_sequence[i] == merchant_sequence[i-1]:
                    reuse_count += 1
            reuse_rate = reuse_count / (len(merchant_sequence) - 1)
            metrics['merchant_reuse_rates'].append(reuse_rate)
        
        # Transaction amounts
        for tx in pattern['legitimate_transactions'] + pattern['fraudulent_transactions']:
            if tx.get('amount'):
                metrics['transaction_amounts'].append(tx['amount'])
    
    print(f"Experiment {run_id} completed: {len(patterns)} patterns found")
    return metrics

def create_validation_plot(all_metrics, config):
    """Create comprehensive validation plot."""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Friendly Fraud Pattern Validation\nMultiple Runs with 500K Transactions', 
                 fontsize=16, fontweight='bold')
    
    # Get config values for reference lines
    expected_legit_tx_min = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['min']
    expected_legit_tx_max = config['fraud_patterns']['friendly_fraud']['initial_legitimate_transactions']['max']
    expected_fraud_tx_min = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['min']
    expected_fraud_tx_max = config['fraud_patterns']['friendly_fraud']['fraudulent_transactions']['max']
    expected_legit_period_min = config['fraud_patterns']['friendly_fraud']['legitimate_period']['min']
    expected_legit_period_max = config['fraud_patterns']['friendly_fraud']['legitimate_period']['max']
    expected_cb_prob = config['fraud_patterns']['friendly_fraud']['chargeback_probability']
    expected_cb_delay_min = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['min']
    expected_cb_delay_max = config['fraud_patterns']['friendly_fraud']['chargeback_delay']['max']
    expected_merchant_reuse = config['fraud_patterns']['friendly_fraud']['merchant_reuse_prob']
    expected_amount_min = config['fraud_patterns']['friendly_fraud']['transaction_amount']['min']
    expected_amount_max = config['fraud_patterns']['friendly_fraud']['transaction_amount']['max']
    
    # Panel 1: Transaction Counts (Legitimate vs Fraudulent)
    ax = axes[0, 0]
    all_legit_counts = []
    all_fraud_counts = []
    for metrics in all_metrics:
        all_legit_counts.extend(metrics['legitimate_tx_counts'])
        all_fraud_counts.extend(metrics['fraudulent_tx_counts'])
    
    x = np.arange(len(all_metrics))
    avg_legit = [np.mean(m['legitimate_tx_counts']) for m in all_metrics]
    avg_fraud = [np.mean(m['fraudulent_tx_counts']) for m in all_metrics]
    
    ax.bar(x - 0.2, avg_legit, 0.4, label='Avg Legitimate', alpha=0.8, color='lightblue')
    ax.bar(x + 0.2, avg_fraud, 0.4, label='Avg Fraudulent', alpha=0.8, color='salmon')
    ax.axhline(expected_legit_tx_min, color='blue', linestyle='--', alpha=0.7, label=f'Legit Min: {expected_legit_tx_min}')
    ax.axhline(expected_legit_tx_max, color='blue', linestyle='--', alpha=0.7, label=f'Legit Max: {expected_legit_tx_max}')
    ax.axhline(expected_fraud_tx_min, color='red', linestyle='--', alpha=0.7, label=f'Fraud Min: {expected_fraud_tx_min}')
    ax.axhline(expected_fraud_tx_max, color='red', linestyle='--', alpha=0.7, label=f'Fraud Max: {expected_fraud_tx_max}')
    ax.set_xlabel('Experiment Run')
    ax.set_ylabel('Average Transaction Count')
    ax.set_title('Transaction Counts per Pattern')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Legitimate Period Distribution
    ax = axes[0, 1]
    all_periods = []
    for metrics in all_metrics:
        all_periods.extend(metrics['legitimate_periods'])
    
    ax.hist(all_periods, bins=20, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    ax.axvline(expected_legit_period_min, color='red', linestyle='--', linewidth=2, 
               label=f'Config Min: {expected_legit_period_min} days')
    ax.axvline(expected_legit_period_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: {expected_legit_period_max} days')
    ax.axvline(np.mean(all_periods), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_periods):.1f} days')
    ax.set_xlabel('Legitimate Period (days)')
    ax.set_ylabel('Density')
    ax.set_title('Legitimate Period Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Chargeback Ratios
    ax = axes[0, 2]
    all_cb_ratios = []
    for metrics in all_metrics:
        all_cb_ratios.extend(metrics['chargeback_ratios'])
    
    ax.hist(all_cb_ratios, bins=20, alpha=0.7, density=True, color='coral', edgecolor='black')
    ax.axvline(expected_cb_prob, color='red', linestyle='--', linewidth=2, 
               label=f'Config: {expected_cb_prob:.1%}')
    ax.axvline(np.mean(all_cb_ratios), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_cb_ratios):.1%}')
    ax.set_xlabel('Chargeback Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Chargeback Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Chargeback Delays
    ax = axes[1, 0]
    all_delays = []
    for metrics in all_metrics:
        all_delays.extend(metrics['chargeback_delays'])
    
    if all_delays:
        ax.hist(all_delays, bins=25, alpha=0.7, density=True, color='gold', edgecolor='black')
        ax.axvline(expected_cb_delay_min, color='red', linestyle='--', linewidth=2, 
                   label=f'Config Min: {expected_cb_delay_min} days')
        ax.axvline(expected_cb_delay_max, color='red', linestyle='--', linewidth=2, 
                   label=f'Config Max: {expected_cb_delay_max} days')
        ax.axvline(np.mean(all_delays), color='blue', linestyle='-', linewidth=2, 
                   label=f'Actual Avg: {np.mean(all_delays):.1f} days')
    ax.set_xlabel('Chargeback Delay (days)')
    ax.set_ylabel('Density')
    ax.set_title('Chargeback Delay Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Merchant Reuse Rates
    ax = axes[1, 1]
    all_reuse_rates = []
    for metrics in all_metrics:
        all_reuse_rates.extend(metrics['merchant_reuse_rates'])
    
    ax.hist(all_reuse_rates, bins=20, alpha=0.7, density=True, color='lightcyan', edgecolor='black')
    ax.axvline(expected_merchant_reuse, color='red', linestyle='--', linewidth=2, 
               label=f'Config: {expected_merchant_reuse:.1%}')
    ax.axvline(np.mean(all_reuse_rates), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_reuse_rates):.1%}')
    ax.set_xlabel('Merchant Reuse Rate')
    ax.set_ylabel('Density')
    ax.set_title('Merchant Reuse Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Transaction Amounts
    ax = axes[1, 2]
    all_amounts = []
    for metrics in all_metrics:
        all_amounts.extend(metrics['transaction_amounts'])
    
    ax.hist(all_amounts, bins=30, alpha=0.7, density=True, color='plum', edgecolor='black')
    ax.axvline(expected_amount_min, color='red', linestyle='--', linewidth=2, 
               label=f'Config Min: €{expected_amount_min}')
    ax.axvline(expected_amount_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: €{expected_amount_max}')
    ax.axvline(np.mean(all_amounts), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: €{np.mean(all_amounts):.0f}')
    ax.set_xlabel('Transaction Amount (€)')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Amount Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('friendly_fraud_validation_multiple_runs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("FRIENDLY FRAUD VALIDATION SUMMARY")
    print("="*80)
    
    total_patterns = sum(len(m['legitimate_tx_counts']) for m in all_metrics)
    print(f"Total Patterns Analyzed: {total_patterns:,}")
    
    all_legit_counts = []
    all_fraud_counts = []
    all_periods = []
    all_cb_ratios = []
    all_delays = []
    all_reuse_rates = []
    all_amounts = []
    
    for metrics in all_metrics:
        all_legit_counts.extend(metrics['legitimate_tx_counts'])
        all_fraud_counts.extend(metrics['fraudulent_tx_counts'])
        all_periods.extend(metrics['legitimate_periods'])
        all_cb_ratios.extend(metrics['chargeback_ratios'])
        all_delays.extend(metrics['chargeback_delays'])
        all_reuse_rates.extend(metrics['merchant_reuse_rates'])
        all_amounts.extend(metrics['transaction_amounts'])
    
    print(f"\nLegitimate Transactions per Pattern:")
    print(f"  Expected: {expected_legit_tx_min}-{expected_legit_tx_max}")
    print(f"  Actual: {np.min(all_legit_counts):.1f}-{np.max(all_legit_counts):.1f} (avg: {np.mean(all_legit_counts):.1f})")
    
    print(f"\nFraudulent Transactions per Pattern:")
    print(f"  Expected: {expected_fraud_tx_min}-{expected_fraud_tx_max}")
    print(f"  Actual: {np.min(all_fraud_counts):.1f}-{np.max(all_fraud_counts):.1f} (avg: {np.mean(all_fraud_counts):.1f})")
    
    print(f"\nLegitimate Period:")
    print(f"  Expected: {expected_legit_period_min}-{expected_legit_period_max} days")
    if all_periods:
        print(f"  Actual: {np.min(all_periods):.1f}-{np.max(all_periods):.1f} days (avg: {np.mean(all_periods):.1f})")
    
    print(f"\nChargeback Probability:")
    print(f"  Expected: {expected_cb_prob:.1%}")
    if all_cb_ratios:
        print(f"  Actual: {np.mean(all_cb_ratios):.1%} (std: {np.std(all_cb_ratios):.1%})")
    
    print(f"\nChargeback Delays:")
    print(f"  Expected: {expected_cb_delay_min}-{expected_cb_delay_max} days")
    if all_delays:
        print(f"  Actual: {np.min(all_delays):.1f}-{np.max(all_delays):.1f} days (avg: {np.mean(all_delays):.1f})")
    
    print(f"\nMerchant Reuse Rate:")
    print(f"  Expected: {expected_merchant_reuse:.1%}")
    if all_reuse_rates:
        print(f"  Actual: {np.mean(all_reuse_rates):.1%} (std: {np.std(all_reuse_rates):.1%})")
    
    print(f"\nTransaction Amounts:")
    print(f"  Expected: €{expected_amount_min}-€{expected_amount_max}")
    if all_amounts:
        print(f"  Actual: €{np.min(all_amounts):.0f}-€{np.max(all_amounts):.0f} (avg: €{np.mean(all_amounts):.0f})")
    
    print("\n" + "="*80)

def main():
    """Run multiple validation experiments."""
    print("Starting Friendly Fraud Validation with Multiple Runs...")
    
    # Load config
    config_manager = ConfigurationManager()
    config = config_manager.get_config()
    
    # Run multiple experiments with 500k transactions each
    num_runs = 3
    num_transactions = 500000
    
    all_metrics = []
    
    for run_id in range(1, num_runs + 1):
        try:
            metrics = run_experiment(num_transactions, run_id)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error in run {run_id}: {e}")
            continue
    
    if all_metrics:
        print(f"\nCompleted {len(all_metrics)} successful runs")
        create_validation_plot(all_metrics, config)
    else:
        print("No successful runs completed!")

if __name__ == "__main__":
    main() 