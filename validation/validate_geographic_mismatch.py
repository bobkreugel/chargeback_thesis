#!/usr/bin/env python3
"""
Geographic Mismatch Pattern Validation Script
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
    
    # Analyze geographic mismatch patterns
    patterns = []
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'customer' and 
            attr.get('fraud_type') == 'geographic_mismatch'):
            
            customer_id = node
            customer_data = attr
            
            # Find related data
            cards = []
            for edge in graph.out_edges(customer_id, data=True):
                if edge[2].get('edge_type') == 'customer_to_card':
                    cards.append(edge[1])
            
            transactions = []
            locations = []
            chargebacks = []
            
            for card_id in cards:
                for edge in graph.out_edges(card_id, data=True):
                    if edge[2].get('edge_type') == 'card_to_transaction':
                        tx_id = edge[1]
                        tx_data = graph.nodes[tx_id]
                        
                        if tx_data.get('is_chargeback', False):
                            chargebacks.append((tx_id, tx_data))
                        else:
                            transactions.append((tx_id, tx_data))
                            
                            for in_edge in graph.in_edges(tx_id, data=True):
                                edge_type = in_edge[2].get('edge_type')
                                if edge_type in ['billing_to_transaction', 'shipping_to_transaction', 'ip_to_transaction']:
                                    loc_id = in_edge[0]
                                    loc_data = graph.nodes[loc_id]
                                    locations.append((loc_id, loc_data, tx_id, edge_type))
            
            if transactions:
                patterns.append({
                    'customer_id': customer_id,
                    'transactions': transactions,
                    'locations': locations,
                    'chargebacks': chargebacks
                })
    
    # Calculate validation metrics
    metrics = {
        'run_id': run_id,
        'num_transactions': num_transactions,
        'num_patterns': len(patterns),
        'transactions_per_pattern': [],
        'regions_per_pattern': [],
        'countries_per_pattern': [],
        'chargeback_ratios': [],
        'transaction_amounts': [],
        'time_windows': [],
        'geographic_mismatches': [],
        'chargeback_delays': []
    }
    
    for pattern in patterns:
        # Transaction count per pattern
        tx_count = len(pattern['transactions'])
        metrics['transactions_per_pattern'].append(tx_count)
        
        # Geographic analysis
        regions = set()
        countries = set()
        geographic_inconsistency = 0
        
        for tx_id, tx_data in pattern['transactions']:
            tx_locations = [loc for loc in pattern['locations'] if loc[2] == tx_id]
            
            if len(tx_locations) >= 3:  # Should have billing, shipping, IP
                tx_regions = set()
                tx_countries = set()
                
                for loc_id, loc_data, _, edge_type in tx_locations:
                    region = loc_data.get('region', 'Unknown')
                    country = loc_data.get('country', 'Unknown')
                    regions.add(region)
                    countries.add(country)
                    tx_regions.add(region)
                    tx_countries.add(country)
                
                # Count geographic mismatch for this transaction
                if len(tx_regions) > 1 or len(tx_countries) > 1:
                    geographic_inconsistency += 1
            
            # Transaction amounts
            amount = tx_data.get('amount', 0)
            metrics['transaction_amounts'].append(amount)
        
        metrics['regions_per_pattern'].append(len(regions))
        metrics['countries_per_pattern'].append(len(countries))
        metrics['geographic_mismatches'].append(geographic_inconsistency / max(tx_count, 1))
        
        # Chargeback ratio
        chargeback_count = len(pattern['chargebacks'])
        chargeback_ratio = chargeback_count / max(tx_count, 1)
        metrics['chargeback_ratios'].append(chargeback_ratio)
        
        # Time window analysis
        if pattern['transactions']:
            timestamps = [tx_data.get('timestamp') for tx_id, tx_data in pattern['transactions'] 
                         if tx_data.get('timestamp')]
            if len(timestamps) > 1:
                timestamps.sort()
                time_window = (timestamps[-1] - timestamps[0]).days
                metrics['time_windows'].append(time_window)
        
        # Chargeback delays
        for cb_id, cb_data in pattern['chargebacks']:
            original_tx = cb_data.get('original_transaction')
            if original_tx:
                for tx_id, tx_data in pattern['transactions']:
                    if tx_id == original_tx:
                        cb_time = cb_data.get('timestamp')
                        tx_time = tx_data.get('timestamp')
                        if cb_time and tx_time:
                            delay = (cb_time - tx_time).days
                            metrics['chargeback_delays'].append(delay)
                        break
    
    print(f"Experiment {run_id} completed: {len(patterns)} patterns found")
    return metrics

def create_validation_plot(all_metrics, config):
    """Create comprehensive validation plot."""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Geographic Mismatch Pattern Validation\nMultiple Runs with 500K Transactions', 
                 fontsize=16, fontweight='bold')
    
    # Get config values for reference lines
    expected_tx_min = config['fraud_patterns']['geographic_mismatch']['num_transactions']['min']
    expected_tx_max = config['fraud_patterns']['geographic_mismatch']['num_transactions']['max']
    expected_amount_min = config['fraud_patterns']['geographic_mismatch']['transaction_amount']['min']
    expected_amount_max = config['fraud_patterns']['geographic_mismatch']['transaction_amount']['max']
    expected_time_min = config['fraud_patterns']['geographic_mismatch']['time_window']['min']
    expected_time_max = config['fraud_patterns']['geographic_mismatch']['time_window']['max']
    expected_cb_prob = config['fraud_patterns']['geographic_mismatch']['chargeback_probability']
    expected_cb_delay_min = config['fraud_patterns']['geographic_mismatch']['chargeback_delay']['min']
    expected_cb_delay_max = config['fraud_patterns']['geographic_mismatch']['chargeback_delay']['max']
    
    # Panel 1: Transactions per Pattern
    ax = axes[0, 0]
    all_tx_counts = []
    for metrics in all_metrics:
        all_tx_counts.extend(metrics['transactions_per_pattern'])
    
    ax.hist(all_tx_counts, bins=range(1, 8), alpha=0.7, density=True, 
            color='skyblue', edgecolor='black')
    ax.axvline(expected_tx_min, color='red', linestyle='--', linewidth=2, label=f'Config Min: {expected_tx_min}')
    ax.axvline(expected_tx_max, color='red', linestyle='--', linewidth=2, label=f'Config Max: {expected_tx_max}')
    ax.set_xlabel('Transactions per Pattern')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Geographic Spread
    ax = axes[0, 1]
    all_regions = []
    all_countries = []
    for metrics in all_metrics:
        all_regions.extend(metrics['regions_per_pattern'])
        all_countries.extend(metrics['countries_per_pattern'])
    
    x = np.arange(len(all_metrics))
    avg_regions = [np.mean(m['regions_per_pattern']) for m in all_metrics]
    avg_countries = [np.mean(m['countries_per_pattern']) for m in all_metrics]
    
    ax.bar(x - 0.2, avg_regions, 0.4, label='Avg Regions', alpha=0.8, color='orange')
    ax.bar(x + 0.2, avg_countries, 0.4, label='Avg Countries', alpha=0.8, color='green')
    ax.set_xlabel('Experiment Run')
    ax.set_ylabel('Average Count')
    ax.set_title('Geographic Diversity per Run')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Transaction Amounts
    ax = axes[0, 2]
    all_amounts = []
    for metrics in all_metrics:
        all_amounts.extend(metrics['transaction_amounts'])
    
    ax.hist(all_amounts, bins=30, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    ax.axvline(expected_amount_min, color='red', linestyle='--', linewidth=2, 
               label=f'Config Min: €{expected_amount_min}')
    ax.axvline(expected_amount_max, color='red', linestyle='--', linewidth=2, 
               label=f'Config Max: €{expected_amount_max}')
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
    
    # Panel 5: Time Windows
    ax = axes[1, 1]
    all_time_windows = []
    for metrics in all_metrics:
        all_time_windows.extend(metrics['time_windows'])
    
    if all_time_windows:
        ax.hist(all_time_windows, bins=range(0, 15), alpha=0.7, density=True, 
                color='lightblue', edgecolor='black')
        ax.axvline(expected_time_min, color='red', linestyle='--', linewidth=2, 
                   label=f'Config Min: {expected_time_min} days')
        ax.axvline(expected_time_max, color='red', linestyle='--', linewidth=2, 
                   label=f'Config Max: {expected_time_max} days')
    ax.set_xlabel('Time Window (days)')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Time Window Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Geographic Mismatch Rate
    ax = axes[1, 2]
    all_mismatches = []
    for metrics in all_metrics:
        all_mismatches.extend(metrics['geographic_mismatches'])
    
    ax.hist(all_mismatches, bins=20, alpha=0.7, density=True, color='gold', edgecolor='black')
    ax.axvline(np.mean(all_mismatches), color='blue', linestyle='-', linewidth=2, 
               label=f'Avg Mismatch Rate: {np.mean(all_mismatches):.1%}')
    ax.set_xlabel('Geographic Mismatch Rate per Pattern')
    ax.set_ylabel('Density')
    ax.set_title('Geographic Inconsistency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('geographic_mismatch_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total_patterns = sum(len(m['transactions_per_pattern']) for m in all_metrics)
    print(f"Total Patterns Analyzed: {total_patterns:,}")
    
    all_tx_counts = []
    all_amounts = []
    all_cb_ratios = []
    all_time_windows = []
    all_regions = []
    all_countries = []
    all_mismatches = []
    
    for metrics in all_metrics:
        all_tx_counts.extend(metrics['transactions_per_pattern'])
        all_amounts.extend(metrics['transaction_amounts'])
        all_cb_ratios.extend(metrics['chargeback_ratios'])
        all_time_windows.extend(metrics['time_windows'])
        all_regions.extend(metrics['regions_per_pattern'])
        all_countries.extend(metrics['countries_per_pattern'])
        all_mismatches.extend(metrics['geographic_mismatches'])
    
    print(f"\nTransaction Count per Pattern:")
    print(f"  Expected: {expected_tx_min}-{expected_tx_max}")
    print(f"  Actual: {np.min(all_tx_counts):.1f}-{np.max(all_tx_counts):.1f} (avg: {np.mean(all_tx_counts):.1f})")
    
    print(f"\nTransaction Amounts:")
    print(f"  Expected: €{expected_amount_min}-€{expected_amount_max}")
    print(f"  Actual: €{np.min(all_amounts):.0f}-€{np.max(all_amounts):.0f} (avg: €{np.mean(all_amounts):.0f})")
    
    print(f"\nChargeback Probability:")
    print(f"  Expected: {expected_cb_prob:.1%}")
    print(f"  Actual: {np.mean(all_cb_ratios):.1%} (std: {np.std(all_cb_ratios):.1%})")
    
    print(f"\nTime Windows:")
    print(f"  Expected: {expected_time_min}-{expected_time_max} days")
    if all_time_windows:
        print(f"  Actual: {np.min(all_time_windows):.1f}-{np.max(all_time_windows):.1f} days (avg: {np.mean(all_time_windows):.1f})")
    
    print(f"\nGeographic Diversity:")
    print(f"  Avg Regions per Pattern: {np.mean(all_regions):.1f}")
    print(f"  Avg Countries per Pattern: {np.mean(all_countries):.1f}")
    print(f"  Geographic Mismatch Rate: {np.mean(all_mismatches):.1%}")
    
    print("\n" + "="*80)

def main():
    """Run multiple validation experiments."""
    print("Starting Geographic Mismatch Validation with Multiple Runs...")
    
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