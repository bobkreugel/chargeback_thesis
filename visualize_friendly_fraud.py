import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from pathlib import Path
import yaml
from datetime import datetime, timedelta

def load_config():
    """Load the configuration file."""
    config_path = Path('src/config/default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_graph(filepath):
    """Load the NetworkX graph from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def find_friendly_fraud_pattern(graph):
    """Find a customer who committed friendly fraud."""
    friendly_fraudsters = []
    
    # Find customers marked as friendly fraudsters
    for node, attrs in graph.nodes(data=True):
        if (attrs.get('node_type') == 'customer' and 
            attrs.get('fraud_type') == 'friendly_fraud'):
            friendly_fraudsters.append(node)
    
    if not friendly_fraudsters:
        print("No friendly fraud patterns found!")
        return None
    
    # Get the first friendly fraudster's subgraph
    customer_id = friendly_fraudsters[0]
    customer_attrs = graph.nodes[customer_id]
    
    # Get all cards for this customer
    cards = []
    for _, card in graph.out_edges(customer_id):
        if graph.nodes[card].get('node_type') == 'card':
            cards.append(card)
    
    if not cards:
        return None
    
    # Get all transactions for the first card
    card_id = cards[0]
    card_attrs = graph.nodes[card_id]
    transactions = []
    merchants = set()
    
    # First collect all original transactions
    original_transactions = {}  # tx_id -> attrs
    chargebacks = []  # list of (chargeback_id, attrs, original_tx)
    
    for _, tx in graph.out_edges(card_id):
        tx_attrs = graph.nodes[tx].copy()
        if tx_attrs.get('node_type') == 'transaction':
            # Find the merchant for this transaction
            for m, t in graph.in_edges(tx):
                if graph.nodes[m].get('node_type') == 'merchant':
                    merchants.add(m)
                    tx_attrs['merchant_id'] = m
                    tx_attrs['merchant_name'] = graph.nodes[m].get('name', 'Unknown Merchant')
                    break
            
            if tx_attrs.get('is_chargeback', False):
                # Get original transaction from node attribute
                original_tx = tx_attrs.get('original_transaction')
                chargebacks.append((tx, tx_attrs, original_tx))
            else:
                original_transactions[tx] = tx_attrs
                transactions.append((tx, tx_attrs))
    
    print(f"\nFound {len(original_transactions)} original transactions and {len(chargebacks)} chargebacks")
    
    # Add chargebacks to transaction list and calculate delays
    for cb_id, cb_attrs, original_tx_id in chargebacks:
        print(f"\nProcessing chargeback {cb_id}:")
        print(f"Original transaction ID: {original_tx_id}")
        
        if original_tx_id and original_tx_id in original_transactions:
            # Calculate delay in days
            original_date = original_transactions[original_tx_id].get('timestamp')
            chargeback_date = cb_attrs.get('timestamp')
            print(f"Original date: {original_date}")
            print(f"Chargeback date: {chargeback_date}")
            
            if original_date and chargeback_date:
                delay_days = (chargeback_date - original_date).days
                cb_attrs['chargeback_delay_days'] = delay_days
                cb_attrs['original_transaction_id'] = original_tx_id
                print(f"Calculated delay: {delay_days} days")
            else:
                print("Missing date information!")
        else:
            print("Could not find original transaction!")
        transactions.append((cb_id, cb_attrs))
    
    # Sort transactions by date
    transactions.sort(key=lambda x: x[1].get('timestamp', datetime.now()))
    
    # Create a new graph with just this pattern
    pattern = nx.DiGraph()
    
    # Add customer with better label
    pattern.add_node(customer_id, 
                    **customer_attrs,
                    pos=(0.5, 1.0),
                    label=f"Friendly Fraud Customer\nID: {customer_id[:8]}")
    
    # Add card with better label
    pattern.add_node(card_id, 
                    **card_attrs,
                    pos=(0.5, 0.8),
                    label=f"Card\nID: {card_id[:8]}")
    pattern.add_edge(customer_id, card_id, edge_type='HAS_CARD', label='HAS CARD')
    
    # Calculate positions for transactions and merchants
    num_tx = len(transactions)
    for i, (tx_id, tx_attrs) in enumerate(transactions):
        # Position transactions in a timeline
        tx_x = (i + 1) / (num_tx + 1)
        tx_y = 0.4
        merchant_y = 0.0
        
        # Add merchant if not already added
        merchant_id = tx_attrs.get('merchant_id')
        if merchant_id and merchant_id not in pattern.nodes():
            pattern.add_node(merchant_id, 
                           **graph.nodes[merchant_id],
                           pos=(tx_x, merchant_y),
                           label=f"Merchant\n{tx_attrs['merchant_name']}")
        
        # Create better transaction label
        date = tx_attrs.get('timestamp', datetime.now())
        amount = tx_attrs.get('amount', 0.0)
        is_chargeback = tx_attrs.get('is_chargeback', False)
        
        if is_chargeback:
            delay_days = tx_attrs.get('chargeback_delay_days', '?')
            label = f"Chargeback\nafter {delay_days} days\n"
            label += f"€{amount:.2f}\n{date.strftime('%Y-%m-%d')}"
        else:
            label = f"Transaction\n€{amount:.2f}\n{date.strftime('%Y-%m-%d')}"
        
        # Add transaction
        pattern.add_node(tx_id, 
                        **tx_attrs,
                        pos=(tx_x, tx_y),
                        label=label)
        
        # Add edges with correct direction
        pattern.add_edge(card_id, tx_id, edge_type='USED_IN', label='USED IN')
        if merchant_id:
            if is_chargeback:
                # For chargebacks, edge goes from merchant to chargeback
                pattern.add_edge(merchant_id, tx_id, 
                               edge_type='GENERATED_CHARGEBACK', 
                               label='Generated Chargeback')
            else:
                # For normal transactions, edge goes from transaction to merchant
                pattern.add_edge(tx_id, merchant_id, 
                               edge_type='TRANSACTION_AT', 
                               label='TRANSACTION AT')
        
        # Add edge from original transaction to chargeback if applicable
        if is_chargeback and 'original_transaction_id' in tx_attrs:
            original_tx_id = tx_attrs['original_transaction_id']
            if original_tx_id in pattern.nodes():
                pattern.add_edge(original_tx_id, tx_id, 
                               edge_type='RESULTED_IN', 
                               label='Resulted in chargeback')
    
    return pattern

def visualize_graph(G):
    """Visualize the friendly fraud pattern graph."""
    plt.figure(figsize=(15, 10))
    
    # Get the position attributes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node]['node_type']
        if node_type == 'customer':
            node_colors.append('lightblue')
            node_sizes.append(3000)
        elif node_type == 'card':
            node_colors.append('lightgreen')
            node_sizes.append(2000)
        elif node_type == 'merchant':
            node_colors.append('lightgray')
            node_sizes.append(2500)
        else:  # transaction
            if G.nodes[node].get('is_chargeback', False):
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgreen')
            node_sizes.append(3500)
    
    # Create a new graph for edges to control what gets drawn
    edge_graph = nx.DiGraph()
    edge_labels = {}
    
    # Add all nodes with their positions
    for node in G.nodes():
        edge_graph.add_node(node, pos=pos[node])
    
    # Add edges with modified logic for chargebacks
    for u, v, data in G.edges(data=True):
        # Get node attributes
        v_attrs = G.nodes[v]
        u_attrs = G.nodes[u]
        
        # Skip edges to chargebacks from cards
        if v_attrs.get('is_chargeback', False) and u_attrs.get('node_type') == 'card':
            continue
            
        # Skip edges from original transactions to chargebacks
        if v_attrs.get('is_chargeback', False) and u_attrs.get('node_type') == 'transaction':
            continue
            
        # For chargebacks, reverse the edge direction from merchant to chargeback
        if v_attrs.get('is_chargeback', False) and u_attrs.get('node_type') == 'merchant':
            # Add edge from merchant to chargeback instead
            merchant_id = u
            chargeback_id = v
            edge_graph.add_edge(merchant_id, chargeback_id)
            if 'label' in data:
                edge_labels[(merchant_id, chargeback_id)] = 'Generated Chargeback'
            continue
            
        # For normal transactions, keep the original direction
        edge_graph.add_edge(u, v)
        if 'label' in data:
            edge_labels[(u, v)] = data['label']
    
    # Draw the graph with the modified edges
    nx.draw(edge_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2,
            with_labels=False)
    
    # Add labels with better positioning
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add edge labels with better positioning
    nx.draw_networkx_edge_labels(edge_graph, pos, edge_labels, font_size=8)
    
    plt.title('Real Friendly Fraud Pattern Example\nShowing progression from legitimate transactions to chargebacks', 
              pad=20)
    plt.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Customer',
                  markerfacecolor='lightblue', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Card',
                  markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Merchant',
                  markerfacecolor='lightgray', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Legitimate Transaction',
                  markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Chargeback',
                  markerfacecolor='lightcoral', markersize=15),
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.tight_layout()
    plt.savefig('friendly_fraud_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Find and extract a friendly fraud pattern
    print("Finding friendly fraud pattern...")
    pattern = find_friendly_fraud_pattern(graph)
    
    if pattern is None:
        print("Could not find a suitable friendly fraud pattern!")
        return
    
    # Create visualization
    print("Creating visualization...")
    visualize_graph(pattern)
    print("Graph visualization saved as 'friendly_fraud_pattern.png'")

if __name__ == "__main__":
    main() 