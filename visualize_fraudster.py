import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
from collections import defaultdict

def load_graph(filepath):
    """Load the NetworkX graph from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_random_fraudster(graph):
    """Get a random fraudster from the graph."""
    fraudsters = [
        node for node, attr in graph.nodes(data=True)
        if attr.get('node_type') == 'customer' and 
        attr.get('fraud_type') == 'serial_chargeback'
    ]
    return random.choice(fraudsters)

def create_fraudster_subgraph(graph, fraudster_id):
    """Create a simplified subgraph showing only customer, card, and merchants."""
    # Get the fraudster's card
    card_edges = [
        (u, v) for u, v, attr in graph.edges(data=True)
        if u == fraudster_id and attr.get('edge_type') == 'customer_to_card'
    ]
    
    # Track merchants and their transaction counts
    merchant_transactions = defaultdict(list)
    merchant_chargebacks = defaultdict(int)
    
    # Get all transactions and count merchant usage
    for _, card_id in card_edges:
        tx_edges = [
            (u, v) for u, v, attr in graph.edges(data=True)
            if u == card_id and attr.get('edge_type') == 'card_to_transaction'
        ]
        
        for _, tx_id in tx_edges:
            # Get merchant for this transaction
            merchant_edges = [
                (u, v) for u, v, attr in graph.edges(data=True)
                if v == tx_id and attr.get('edge_type') == 'merchant_to_transaction'
            ]
            
            for merchant_id, _ in merchant_edges:
                # Store transaction info
                is_chargeback = graph.nodes[tx_id].get('is_chargeback', False)
                amount = graph.nodes[tx_id].get('amount', 0)
                timestamp = graph.nodes[tx_id].get('timestamp')
                
                merchant_transactions[merchant_id].append({
                    'amount': amount,
                    'timestamp': timestamp,
                    'is_chargeback': is_chargeback
                })
                
                if is_chargeback:
                    merchant_chargebacks[merchant_id] += 1
    
    # Create new graph with just customer, card, and merchants
    simplified_graph = nx.DiGraph()
    
    # Add customer node
    customer_data = graph.nodes[fraudster_id]
    simplified_graph.add_node(
        fraudster_id,
        node_type='customer',
        name=customer_data.get('name', ''),
        fraud_type='serial_chargeback'
    )
    
    # Add card node and edge
    for customer_id, card_id in card_edges:
        card_data = graph.nodes[card_id]
        simplified_graph.add_node(
            card_id,
            node_type='card',
            card_type=card_data.get('card_type', '')
        )
        simplified_graph.add_edge(customer_id, card_id, edge_type='customer_to_card')
    
    # Add merchant nodes and edges
    for merchant_id, transactions in merchant_transactions.items():
        merchant_data = graph.nodes[merchant_id]
        num_tx = len([tx for tx in transactions if not tx['is_chargeback']])
        num_cb = merchant_chargebacks[merchant_id]
        
        simplified_graph.add_node(
            merchant_id,
            node_type='merchant',
            name=merchant_data.get('name', ''),
            num_transactions=num_tx,
            num_chargebacks=num_cb
        )
        
        # Add edge from card to merchant
        for _, card_id in card_edges:
            simplified_graph.add_edge(
                card_id,
                merchant_id,
                edge_type='card_to_merchant',
                has_chargebacks=(num_cb > 0)
            )
    
    return simplified_graph

def visualize_fraudster_pattern(graph, fraudster_id):
    """Create a visualization of the fraudster's transaction pattern."""
    # Create simplified subgraph
    subgraph = create_fraudster_subgraph(graph, fraudster_id)
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    labels = {}
    
    for node in subgraph.nodes():
        attr = subgraph.nodes[node]
        node_type = attr.get('node_type')
        
        if node_type == 'customer':
            node_colors.append('lightblue')
            node_sizes.append(2000)
            labels[node] = f"Fraudster\n{attr.get('name', '')[:20]}"
        elif node_type == 'card':
            node_colors.append('lightgreen')
            node_sizes.append(1500)
            labels[node] = f"Card\n{attr.get('card_type', '')}"
        elif node_type == 'merchant':
            node_colors.append('orange')
            node_sizes.append(1500)
            num_tx = attr.get('num_transactions', 0)
            num_cb = attr.get('num_chargebacks', 0)
            num_normal = num_tx - num_cb  # Calculate normal transactions
            labels[node] = f"Merchant\n{attr.get('name', '')[:20]}\n{num_normal} normal, {num_cb} chargebacks"
    
    # Draw the network
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes)
    
    # Draw edges with different colors and widths based on type and transaction count
    normal_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                   if d.get('edge_type') == 'customer_to_card']
    
    # Get merchant edges with their transaction counts
    merchant_edges = []
    merchant_widths = []
    merchant_colors = []
    
    for u, v, d in subgraph.edges(data=True):
        if d.get('edge_type') == 'card_to_merchant':
            merchant = v
            num_tx = subgraph.nodes[merchant]['num_transactions']
            num_cb = subgraph.nodes[merchant]['num_chargebacks']
            num_normal = num_tx - num_cb  # Calculate normal transactions
            
            # Add edge for normal transactions
            if num_normal > 0:
                merchant_edges.append((u, v))
                merchant_widths.append(1 + num_normal)  # Base width on normal transaction count
                merchant_colors.append('lightblue')  # Use lightblue for normal transactions
            
            # Add edge for chargeback transactions
            if num_cb > 0:
                merchant_edges.append((u, v))
                merchant_widths.append(1 + num_cb)  # Base width on chargeback count
                merchant_colors.append('red')
    
    # Draw edges in layers
    # First the customer-to-card edge
    nx.draw_networkx_edges(subgraph, pos, edgelist=normal_edges, 
                          edge_color='gray', arrows=True, arrowsize=20)
    
    # Then the merchant edges with varying widths
    for edge, width, color in zip(merchant_edges, merchant_widths, merchant_colors):
        nx.draw_networkx_edges(subgraph, pos, edgelist=[edge],
                             edge_color=color, arrows=True, 
                             width=width, arrowsize=20)
    
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    # Add title with statistics
    merchant_nodes = [n for n, attr in subgraph.nodes(data=True) if attr.get('node_type') == 'merchant']
    merchants_with_cb = [n for n in merchant_nodes if subgraph.nodes[n]['num_chargebacks'] > 0]
    total_tx = sum(subgraph.nodes[n]['num_transactions'] for n in merchant_nodes)
    total_cb = sum(subgraph.nodes[n]['num_chargebacks'] for n in merchant_nodes)
    total_normal = total_tx - total_cb
    
    title = f"Serial Chargeback Pattern Structure\n"
    title += f"Unique Merchants: {len(merchant_nodes)} "
    title += f"(with {len(merchants_with_cb)} having chargebacks)\n"
    title += f"Total Transactions: {total_normal} normal (blue), {total_cb} chargebacks (red)"
    plt.title(title)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='lightblue', label='Normal Transactions'),
        plt.Line2D([0], [0], color='red', label='Chargebacks'),
        plt.Line2D([0], [0], marker='o', color='w', label='Fraudster',
                  markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Card',
                  markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Merchant',
                  markerfacecolor='orange', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot with extra space for legend
    plt.savefig('fraudster_pattern.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the graph
    graph = load_graph('output/dataset/transaction_graph.gpickle')
    
    # Get a random fraudster
    fraudster_id = get_random_fraudster(graph)
    
    # Create visualization
    visualize_fraudster_pattern(graph, fraudster_id)
    print("Visualization saved as 'fraudster_pattern.png'")

if __name__ == "__main__":
    main() 