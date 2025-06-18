#!/usr/bin/env python3
"""
Serial Chargeback Pattern Validation Script
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
    
    # Analyze serial chargeback patterns
    patterns = []
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'customer' and 
            attr.get('fraud_type') == 'serial_chargeback'):
            
            customer_id = node
            customer_data = attr
            
            # Find customer's card
            card_id = None
            for edge in graph.out_edges(customer_id, data=True):
                if edge[2].get('edge_type') == 'customer_to_card':
                    card_id = edge[1]
                    break
            
            if not card_id:
                continue
            
            transactions = []
            chargebacks = []
            merchants_used = set()
            
            # Get all transactions for this card
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
                patterns.append({
                    'customer_id': customer_id,
                    'transactions': transactions,
                    'chargebacks': chargebacks,
                    'merchants_used': merchants_used,
                    'card_id': card_id
                })
    
    # Calculate validation metrics
    metrics = {
        'run_id': run_id,
        'num_transactions': num_transactions,
        'num_patterns': len(patterns),
        'transactions_per_pattern': [],
        'time_windows': [],  # in days
        'transaction_amounts': [],
        'chargeback_ratios': [],
        'chargeback_delays': [],
        'merchant_reuse_rates': [],
        'pattern_durations': [],  # total pattern duration in days
        'merchants_per_pattern': []
    }
    
    for pattern in patterns:
        # Transactions per pattern
        num_transactions = len(pattern['transactions'])
        metrics['transactions_per_pattern'].append(num_transactions)
        
        # Merchants per pattern
        num_merchants = len(pattern['merchants_used'])
        metrics['merchants_per_pattern'].append(num_merchants)
        
        # Time window analysis
        if pattern['transactions']:
            timestamps = [tx['timestamp'] for tx in pattern['transactions'] if tx.get('timestamp')]
            if len(timestamps) > 1:
                timestamps.sort()
                # Total pattern duration
                total_duration = (timestamps[-1] - timestamps[0]).days
                metrics['pattern_durations'].append(total_duration)
                
                # Time windows between consecutive transactions
                for i in range(1, len(timestamps)):
                    time_diff = (timestamps[i] - timestamps[i-1]).days
                    metrics['time_windows'].append(time_diff)
            else:
                metrics['pattern_durations'].append(0)
        
        # Transaction amounts
        for tx in pattern['transactions']:
            if tx.get('amount'):
                metrics['transaction_amounts'].append(tx['amount'])
        
        # Chargeback analysis
        total_txs = len(pattern['transactions'])
        total_chargebacks = len(pattern['chargebacks'])
        if total_txs > 0:
            chargeback_ratio = total_chargebacks / total_txs
            metrics['chargeback_ratios'].append(chargeback_ratio)
        
        # Chargeback delays
        for chargeback in pattern['chargebacks']:
            original_tx_id = chargeback.get('original_transaction')
            if original_tx_id:
                for tx in pattern['transactions']:
                    if tx['transaction_id'] == original_tx_id:
                        if chargeback.get('timestamp') and tx.get('timestamp'):
                            delay = (chargeback['timestamp'] - tx['timestamp']).days
                            metrics['chargeback_delays'].append(delay)
                        break
        
        # Merchant reuse analysis - calculate actual reuse rate
        if total_txs > 1:
            merchant_sequence = [tx['merchant_id'] for tx in pattern['transactions']]
            reuse_count = 0
            for i in range(1, len(merchant_sequence)):
                if merchant_sequence[i] == merchant_sequence[i-1]:
                    reuse_count += 1
            reuse_rate = reuse_count / (total_txs - 1) if total_txs > 1 else 0
            metrics['merchant_reuse_rates'].append(reuse_rate)
    
    print(f"Experiment {run_id} completed: {len(patterns)} patterns found")
    return metrics

def create_validation_plot(all_metrics, config):
    """Create comprehensive validation plot."""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Serial Chargeback Pattern Validation\nMultiple Runs with 500K Transactions', 
                 fontsize=16, fontweight='bold')
    
    # Get config values for reference lines
    expected_tx_min = config['fraud_patterns']['serial_chargeback']['transactions_in_pattern']['min']
    expected_tx_max = config['fraud_patterns']['serial_chargeback']['transactions_in_pattern']['max']
    expected_time_window_min = config['fraud_patterns']['serial_chargeback']['time_window']['min']
    expected_time_window_max = config['fraud_patterns']['serial_chargeback']['time_window']['max']
    expected_amount_min = config['transactions']['amount']['min']  # Uses global transaction amounts
    expected_amount_max = config['transactions']['amount']['max']  # Uses global transaction amounts
    expected_merchant_reuse = config['fraud_patterns']['serial_chargeback']['merchant_reuse_prob']
    expected_chargeback_rate = config['fraud_patterns']['serial_chargeback']['chargeback_probability']  # Now 80% instead of 100%
    expected_cb_delay_min = config['fraud_patterns']['serial_chargeback']['chargeback_delay']['min']
    expected_cb_delay_max = config['fraud_patterns']['serial_chargeback']['chargeback_delay']['max']
    
    # Panel 1: Transactions per Pattern
    ax = axes[0, 0]
    all_tx_counts = []
    for metrics in all_metrics:
        all_tx_counts.extend(metrics['transactions_per_pattern'])
    
    ax.hist(all_tx_counts, bins=range(expected_tx_min, expected_tx_max + 2), 
            alpha=0.7, density=True, color='lightblue', edgecolor='black', align='left')
    ax.axvline(expected_tx_min, color='red', linestyle='--', linewidth=2, 
               label=f'Config Min: {expected_tx_min}')
    ax.axvline(expected_tx_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: {expected_tx_max}')
    ax.axvline(np.mean(all_tx_counts), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_tx_counts):.1f}')
    ax.set_xlabel('Transactions per Pattern')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Pattern Duration Distribution
    ax = axes[0, 1]
    all_durations = []
    for metrics in all_metrics:
        all_durations.extend(metrics['pattern_durations'])
    
    ax.hist(all_durations, bins=25, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    ax.axvline(expected_time_window_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: {expected_time_window_max} days')
    ax.axvline(np.mean(all_durations), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_durations):.1f} days')
    ax.set_xlabel('Pattern Duration (days)')
    ax.set_ylabel('Density')
    ax.set_title('Serial Chargeback Pattern Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Transaction Amounts
    ax = axes[0, 2]
    all_amounts = []
    for metrics in all_metrics:
        all_amounts.extend(metrics['transaction_amounts'])
    
    ax.hist(all_amounts, bins=20, alpha=0.7, density=True, color='coral', edgecolor='black')
    ax.axvline(expected_amount_min, color='red', linestyle='--', linewidth=2, 
               label=f'Config Min: €{expected_amount_min}')
    ax.axvline(expected_amount_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: €{expected_amount_max}')
    ax.axvline(np.mean(all_amounts), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: €{np.mean(all_amounts):.2f}')
    ax.set_xlabel('Transaction Amount (€)')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Amount Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Chargeback Ratios
    ax = axes[1, 0]
    all_cb_ratios = []
    for metrics in all_metrics:
        all_cb_ratios.extend(metrics['chargeback_ratios'])
    
    ax.hist(all_cb_ratios, bins=20, alpha=0.7, density=True, color='gold', edgecolor='black')
    ax.axvline(expected_chargeback_rate, color='red', linestyle='--', linewidth=2, 
               label=f'Config: {expected_chargeback_rate:.1%}')
    ax.axvline(np.mean(all_cb_ratios), color='blue', linestyle='-', linewidth=2, 
               label=f'Actual Avg: {np.mean(all_cb_ratios):.1%}')
    ax.set_xlabel('Chargeback Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Chargeback Rate Distribution')
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
    
    # Panel 6: Chargeback Delays
    ax = axes[1, 2]
    all_delays = []
    for metrics in all_metrics:
        all_delays.extend(metrics['chargeback_delays'])
    
    if all_delays:
        ax.hist(all_delays, bins=20, alpha=0.7, density=True, color='plum', edgecolor='black')
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
    
    plt.tight_layout()
    plt.savefig('serial_chargeback_validation_multiple_runs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SERIAL CHARGEBACK VALIDATION SUMMARY")
    print("="*80)
    
    total_patterns = sum(len(m['transactions_per_pattern']) for m in all_metrics)
    print(f"Total Patterns Analyzed: {total_patterns:,}")
    
    all_tx_counts = []
    all_durations = []
    all_amounts = []
    all_cb_ratios = []
    all_delays = []
    all_reuse_rates = []
    all_merchants = []
    
    for metrics in all_metrics:
        all_tx_counts.extend(metrics['transactions_per_pattern'])
        all_durations.extend(metrics['pattern_durations'])
        all_amounts.extend(metrics['transaction_amounts'])
        all_cb_ratios.extend(metrics['chargeback_ratios'])
        all_delays.extend(metrics['chargeback_delays'])
        all_reuse_rates.extend(metrics['merchant_reuse_rates'])
        all_merchants.extend(metrics['merchants_per_pattern'])
    
    print(f"\nTransactions per Pattern:")
    print(f"  Expected: {expected_tx_min}-{expected_tx_max}")
    print(f"  Actual: {np.min(all_tx_counts):.1f}-{np.max(all_tx_counts):.1f} (avg: {np.mean(all_tx_counts):.1f})")
    
    print(f"\nPattern Duration:")
    print(f"  Expected: {expected_time_window_min}-{expected_time_window_max} days")
    if all_durations:
        print(f"  Actual: {np.min(all_durations):.1f}-{np.max(all_durations):.1f} days (avg: {np.mean(all_durations):.1f})")
    
    print(f"\nTransaction Amounts:")
    print(f"  Expected: €{expected_amount_min}-€{expected_amount_max}")
    if all_amounts:
        print(f"  Actual: €{np.min(all_amounts):.2f}-€{np.max(all_amounts):.2f} (avg: €{np.mean(all_amounts):.2f})")
    
    print(f"\nChargeback Rate:")
    print(f"  Expected: {expected_chargeback_rate:.1%}")
    if all_cb_ratios:
        print(f"  Actual: {np.mean(all_cb_ratios):.1%} (std: {np.std(all_cb_ratios):.1%})")
    
    print(f"\nMerchant Reuse Rate:")
    print(f"  Expected: {expected_merchant_reuse:.1%}")
    if all_reuse_rates:
        print(f"  Actual: {np.mean(all_reuse_rates):.1%} (std: {np.std(all_reuse_rates):.1%})")
    
    print(f"\nChargeback Delays:")
    print(f"  Expected: {expected_cb_delay_min}-{expected_cb_delay_max} days")
    if all_delays:
        print(f"  Actual: {np.min(all_delays):.1f}-{np.max(all_delays):.1f} days (avg: {np.mean(all_delays):.1f})")
    
    print(f"\nMerchants per Pattern:")
    if all_merchants:
        print(f"  Actual: {np.min(all_merchants):.1f}-{np.max(all_merchants):.1f} (avg: {np.mean(all_merchants):.1f})")
    
    print("\n" + "="*80)

def main():
    """Run multiple validation experiments."""
    print("Starting Serial Chargeback Validation with Multiple Runs...")
    
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