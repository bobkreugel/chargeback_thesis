import networkx as nx
import pickle
import os
import statistics
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def load_graph(filepath: str) -> nx.DiGraph:
    """Load the graph from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_config(filepath: str) -> dict:
    """Load the configuration used to generate the data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_fraudster_transactions(graph: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Find all serial chargeback fraudsters and their normal transactions.
    Returns a dictionary mapping customer IDs to lists of their normal transaction IDs.
    """
    fraudster_transactions = {}
    
    # Find all serial chargeback fraudsters
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'customer' and 
            attr.get('is_fraudster', False) and
            attr.get('fraud_type') == 'serial_chargeback'):
            # Get all cards for this customer
            customer_cards = []
            for _, card_id, edge_data in graph.out_edges(node, data=True):
                if edge_data.get('edge_type') == 'customer_to_card':
                    customer_cards.append(card_id)
            
            # Get all normal transactions for these cards
            normal_transactions = []
            for card_id in customer_cards:
                for _, tx_id, edge_data in graph.out_edges(card_id, data=True):
                    if (edge_data.get('edge_type') == 'card_to_transaction' and
                        not graph.nodes[tx_id].get('is_chargeback', False)):
                        normal_transactions.append(tx_id)
            
            fraudster_transactions[node] = normal_transactions
    
    return fraudster_transactions

def get_transaction_details(graph: nx.DiGraph, tx_id: str) -> Tuple[datetime, str, float]:
    """Get timestamp, merchant_id, and amount for a transaction."""
    tx_data = graph.nodes[tx_id]
    timestamp = tx_data['timestamp']
    amount = tx_data['amount']
    
    # Find merchant for this transaction
    merchant_edges = [e for e in graph.in_edges(tx_id, data=True) 
                     if e[2].get('edge_type') == 'merchant_to_transaction']
    merchant_id = merchant_edges[0][0] if merchant_edges else None
    
    return timestamp, merchant_id, amount

def get_chargeback_details(graph: nx.DiGraph, tx_id: str) -> Tuple[str, datetime]:
    """Get chargeback ID and timestamp for a transaction if it exists."""
    # Find chargeback that references this transaction
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'transaction' and 
            attr.get('is_chargeback', False) and 
            attr.get('original_transaction') == tx_id):
            return node, attr['timestamp']
    return None, None

def validate_time_windows(graph: nx.DiGraph, fraudster_transactions: Dict[str, List[str]], 
                        config: dict) -> Tuple[bool, List[float]]:
    """
    Validate that normal transactions (excluding chargebacks) for each fraudster fall within the configured time window.
    The time window only applies to the normal transactions, chargebacks can happen later.
    Returns (is_valid, window_sizes_days).
    """
    window_config = config['fraud_patterns']['serial_chargeback']['time_window']
    min_days = window_config['min']
    max_days = window_config['max']
    window_sizes = []
    all_valid = True
    
    print(f"\nTime Window Validation:")
    print(f"Configuration settings:")
    print(f"- Required window size for normal transactions: {min_days} to {max_days} days")
    print(f"Note: Chargebacks can occur outside this window")
    print(f"\nActual results:")
    
    for customer_id, transactions in fraudster_transactions.items():
        if not transactions:
            continue
            
        # Get timestamps for normal transactions only
        timestamps = []
        for tx_id in transactions:
            # Skip if this is a chargeback
            if graph.nodes[tx_id].get('is_chargeback', False):
                continue
            timestamps.append(get_transaction_details(graph, tx_id)[0])
        
        if timestamps:  # Only calculate window if we have normal transactions
            window_size = (max(timestamps) - min(timestamps)).days
            window_sizes.append(window_size)
            
            if not (min_days <= window_size <= max_days):
                all_valid = False
                print(f"Warning: Customer {customer_id} has normal transactions spread over {window_size} days")
    
    if window_sizes:
        avg_window = statistics.mean(window_sizes)
        print(f"- Average window size for normal transactions: {avg_window:.1f} days")
        print(f"- Window size range: {min(window_sizes):.1f} to {max(window_sizes):.1f} days")
        print(f"All normal transaction windows within range ({min_days}-{max_days} days): {'✓' if all_valid else '✗'}")
    else:
        print("Warning: No normal transactions found!")
        all_valid = False
    
    return all_valid, window_sizes

def validate_merchant_reuse(graph: nx.DiGraph, fraudster_transactions: Dict[str, List[str]], 
                          config: dict) -> Tuple[bool, List[float]]:
    """
    Validate merchant reuse rates for each fraudster individually.
    Returns (is_valid, reuse_rates).
    """
    reuse_config = config['fraud_patterns']['serial_chargeback']['merchant_reuse_prob']
    target_rate = reuse_config
    tolerance = 0.1  # Allow 10% deviation from target
    
    print(f"\nMerchant Reuse Validation:")
    print(f"Configuration settings:")
    print(f"- Target reuse rate: {target_rate:.1%}")
    print(f"\nActual results:")
    
    # Calculate reuse rate for each fraudster
    individual_rates = []
    all_valid = True
    
    for customer_id, transactions in fraudster_transactions.items():
        if len(transactions) < 2:
            continue
            
        # Get merchants for this fraudster's transactions
        merchants = [get_transaction_details(graph, tx_id)[1] for tx_id in transactions]
        merchant_counts = Counter(merchants)
        
        # Calculate reuse rate for this fraudster
        total_txs = len(merchants)
        reused_txs = sum(count - 1 for count in merchant_counts.values())  # Count transactions with reused merchants
        reuse_rate = reused_txs / total_txs if total_txs > 0 else 0
        
        individual_rates.append(reuse_rate)
        
        # Check if this fraudster's rate is within tolerance
        if abs(reuse_rate - target_rate) > tolerance:
            all_valid = False
            print(f"Warning: Customer {customer_id} has merchant reuse rate of {reuse_rate:.1%}")
            print(f"  - Total transactions: {total_txs}")
            print(f"  - Unique merchants: {len(merchant_counts)}")
            print(f"  - Reused transactions: {reused_txs}")
    
    # Print statistics
    if individual_rates:
        avg_rate = statistics.mean(individual_rates)
        print(f"\nPer-fraudster Statistics:")
        print(f"- Average reuse rate: {avg_rate:.1%}")
        print(f"- Rate range: {min(individual_rates):.1%} to {max(individual_rates):.1%}")
        print(f"- Number of fraudsters: {len(individual_rates)}")
        print(f"\nAll reuse rates within ±10% of target ({target_rate:.1%}): {'✓' if all_valid else '✗'}")
    else:
        print("Warning: No valid fraudsters found for merchant reuse calculation!")
        all_valid = False
    
    return all_valid, individual_rates

def validate_chargeback_delays(graph: nx.DiGraph, fraudster_transactions: Dict[str, List[str]], 
                             config: dict) -> Tuple[bool, List[int]]:
    """
    Validate chargeback delays for each transaction.
    Returns (is_valid, delay_days).
    """
    delay_config = config['fraud_patterns']['serial_chargeback']['chargeback_delay']
    min_days = delay_config['min']
    max_days = delay_config['max']
    delay_days = []
    all_valid = True
    
    print(f"\nChargeback Delay Validation:")
    print(f"Configuration settings:")
    print(f"- Required delay: {min_days} to {max_days} days")
    print(f"\nActual results:")
    
    for transactions in fraudster_transactions.values():
        for tx_id in transactions:
            cb_id, cb_timestamp = get_chargeback_details(graph, tx_id)
            if cb_id:
                tx_timestamp = get_transaction_details(graph, tx_id)[0]
                delay = (cb_timestamp - tx_timestamp).days
                delay_days.append(delay)
                
                if not (min_days <= delay <= max_days):
                    all_valid = False
                    print(f"Warning: Transaction {tx_id} has {delay} days chargeback delay")
    
    if delay_days:
        avg_delay = statistics.mean(delay_days)
        print(f"- Average delay: {avg_delay:.1f} days")
        print(f"- Delay range: {min(delay_days)} to {max(delay_days)} days")
        print(f"All delays within range ({min_days}-{max_days} days): {'✓' if all_valid else '✗'}")
    else:
        print("Warning: No chargebacks found!")
        all_valid = False
    
    return all_valid, delay_days

def plot_validation_results(window_sizes: List[float], reuse_rates: List[float], 
                          delay_days: List[int], config: dict) -> None:
    """Create plots showing the validation results."""
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Time Window Distribution
    plt.subplot(3, 1, 1)
    window_config = config['fraud_patterns']['serial_chargeback']['time_window']
    plt.hist(window_sizes, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=window_config['min'], color='g', linestyle='--', label='Min Required')
    plt.axvline(x=window_config['max'], color='g', linestyle='--', label='Max Required')
    plt.axvline(x=statistics.mean(window_sizes), color='r', linestyle='--', label='Average')
    plt.xlabel('Time Window (days)')
    plt.ylabel('Number of Fraudsters')
    plt.title('Distribution of Transaction Time Windows')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Merchant Reuse Distribution
    plt.subplot(3, 1, 2)
    reuse_config = config['fraud_patterns']['serial_chargeback']['merchant_reuse_prob']
    plt.hist(reuse_rates, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=reuse_config, color='g', linestyle='--', label='Target Rate')
    plt.axvline(x=statistics.mean(reuse_rates), color='r', linestyle='--', label='Average')
    plt.xlabel('Merchant Reuse Rate')
    plt.ylabel('Number of Fraudsters')
    plt.title('Distribution of Merchant Reuse Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Chargeback Delay Distribution
    plt.subplot(3, 1, 3)
    delay_config = config['fraud_patterns']['serial_chargeback']['chargeback_delay']
    plt.hist(delay_days, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=delay_config['min'], color='g', linestyle='--', label='Min Required')
    plt.axvline(x=delay_config['max'], color='g', linestyle='--', label='Max Required')
    plt.axvline(x=statistics.mean(delay_days), color='r', linestyle='--', label='Average')
    plt.xlabel('Chargeback Delay (days)')
    plt.ylabel('Number of Transactions')
    plt.title('Distribution of Chargeback Delays')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('serial_chargeback_validation.png')
    plt.close()

def main():
    # Load the graph
    graph_path = "output/dataset/transaction_graph.gpickle"
    config_path = "output/dataset/config.json"
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    print("Loading transaction graph and configuration...")
    graph = load_graph(graph_path)
    config = load_config(config_path)
    
    print("Analyzing fraudulent customer transactions...")
    fraudster_transactions = get_fraudster_transactions(graph)
    
    # Run all validations
    is_valid_windows, window_sizes = validate_time_windows(graph, fraudster_transactions, config)
    is_valid_reuse, reuse_rates = validate_merchant_reuse(graph, fraudster_transactions, config)
    is_valid_delays, delay_days = validate_chargeback_delays(graph, fraudster_transactions, config)
    
    # Create combined visualization
    plot_validation_results(window_sizes, reuse_rates, delay_days, config)
    print("\nVisualization saved as 'serial_chargeback_validation.png'")
    
    # Overall validation result
    all_valid = all([
        is_valid_windows,
        is_valid_reuse,
        is_valid_delays
    ])
    
    if all_valid:
        print("\nAll validation checks passed! ✓")
    else:
        print("\nSome validation checks failed! ✗")

if __name__ == "__main__":
    main() 