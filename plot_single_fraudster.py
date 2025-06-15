import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import random
from matplotlib.patches import Rectangle

def load_graph(filepath: str) -> nx.DiGraph:
    """Load the graph from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_config(filepath: str) -> dict:
    """Load the configuration used to generate the data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_single_fraudster_data(graph: nx.DiGraph) -> Tuple[str, List[Tuple[datetime, float, str, datetime]]]:
    """
    Find a single serial chargeback fraudster and get their transaction data.
    Returns: (customer_id, [(tx_timestamp, amount, tx_id, chargeback_timestamp)])
    """
    # Find all serial chargeback fraudsters
    fraudsters = []
    for node, attr in graph.nodes(data=True):
        if (attr.get('node_type') == 'customer' and 
            attr.get('is_fraudster', False) and
            attr.get('fraud_type') == 'serial_chargeback'):
            fraudsters.append(node)
    
    if not fraudsters:
        raise ValueError("No serial chargeback fraudsters found!")
    
    # Select a random fraudster
    customer_id = random.choice(fraudsters)
    
    # Get all transactions for this customer
    transactions = []
    
    # Get all cards for this customer
    customer_cards = []
    for _, card_id, edge_data in graph.out_edges(customer_id, data=True):
        if edge_data.get('edge_type') == 'customer_to_card':
            customer_cards.append(card_id)
    
    # Get all normal transactions and their chargebacks
    for card_id in customer_cards:
        for _, tx_id, edge_data in graph.out_edges(card_id, data=True):
            if (edge_data.get('edge_type') == 'card_to_transaction' and
                not graph.nodes[tx_id].get('is_chargeback', False)):
                
                tx_data = graph.nodes[tx_id]
                tx_timestamp = tx_data['timestamp']
                amount = tx_data['amount']
                
                # Find corresponding chargeback
                chargeback_timestamp = None
                for node, attr in graph.nodes(data=True):
                    if (attr.get('node_type') == 'transaction' and 
                        attr.get('is_chargeback', False) and 
                        attr.get('original_transaction') == tx_id):
                        chargeback_timestamp = attr['timestamp']
                        break
                
                transactions.append((tx_timestamp, amount, tx_id, chargeback_timestamp))
    
    return customer_id, sorted(transactions, key=lambda x: x[0])

def plot_fraudster_timeline(customer_id: str, transactions: List[Tuple[datetime, float, str, datetime]], 
                          config: dict) -> None:
    """Create a timeline visualization for a single fraudster."""
    # Get configuration values
    window_config = config['fraud_patterns']['serial_chargeback']['time_window']
    delay_config = config['fraud_patterns']['serial_chargeback']['chargeback_delay']
    
    # Create figure and axis
    plt.figure(figsize=(15, 8))
    
    # Plot transactions and chargebacks
    tx_times = [tx[0] for tx in transactions]
    cb_times = [tx[3] for tx in transactions if tx[3] is not None]
    amounts = [tx[1] for tx in transactions]
    max_amount = max(amounts)
    
    # Calculate overall time range
    min_time = min(tx_times)
    max_time = max(cb_times) if cb_times else max(tx_times)
    time_range = max_time - min_time
    
    # Plot transaction window
    window_start = min(tx_times)
    window_end = max(tx_times)
    plt.axvspan(window_start, window_end, alpha=0.2, color='blue', label='Transaction Window')
    
    # Plot chargeback delay windows for each transaction
    for tx_time in tx_times:
        delay_start = tx_time + timedelta(days=delay_config['min'])
        delay_end = tx_time + timedelta(days=delay_config['max'])
        plt.axvspan(delay_start, delay_end, alpha=0.1, color='red')
    
    # Plot transactions
    for i, (tx_time, amount, tx_id, cb_time) in enumerate(transactions):
        # Plot transaction point
        plt.scatter(tx_time, amount, color='blue', s=100, zorder=5)
        
        # Plot chargeback point and connection line if exists
        if cb_time:
            plt.scatter(cb_time, amount, color='red', marker='x', s=100, zorder=5)
            plt.plot([tx_time, cb_time], [amount, amount], 'r--', alpha=0.5)
    
    # Customize the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.title(f'Transaction Timeline for Serial Chargeback Fraudster\nCustomer ID: {customer_id[:8]}...')
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount ($)')
    
    # Add legend with custom patches
    legend_elements = [
        plt.scatter([], [], color='blue', s=100, label='Original Transaction'),
        plt.scatter([], [], color='red', marker='x', s=100, label='Chargeback'),
        plt.plot([], [], 'r--', alpha=0.5, label='Chargeback Connection')[0],
        Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.2, label='Transaction Window'),
        Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.1, label='Chargeback Window')
    ]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    plt.text(1.05, 0.7, 
             f'Configuration:\n\n'
             f'Transaction Window:\n'
             f'  {window_config["min"]} to {window_config["max"]} days\n\n'
             f'Chargeback Delay:\n'
             f'  {delay_config["min"]} to {delay_config["max"]} days\n\n'
             f'Actual Window Size:\n'
             f'  {(window_end - window_start).days} days\n\n'
             f'Number of Transactions:\n'
             f'  {len(transactions)}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('single_fraudster_timeline.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the graph and config
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
    
    print("Finding a serial chargeback fraudster...")
    customer_id, transactions = get_single_fraudster_data(graph)
    
    print(f"Creating visualization for customer {customer_id[:8]}...")
    plot_fraudster_timeline(customer_id, transactions, config)
    print("\nVisualization saved as 'single_fraudster_timeline.png'")

if __name__ == "__main__":
    main() 