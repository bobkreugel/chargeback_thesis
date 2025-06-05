import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Read the data
transactions_df = pd.read_csv('output/dataset/transactions.csv')
cards_df = pd.read_csv('output/dataset/cards.csv')

# Merge transactions with cards
merged_df = transactions_df.merge(cards_df, on='card_id')

# Filter for fraudulent transactions with small amounts (BIN attack characteristic)
bin_attacks = merged_df[
    (merged_df['is_fraudulent'] == True) & 
    (merged_df['amount'] <= 5.0)
].copy()

# Group by source_ip to find an IP with multiple cards (characteristic of BIN attack)
ip_card_counts = bin_attacks.groupby('source_ip')['card_id'].nunique()
target_ip = ip_card_counts.sort_values(ascending=False).index[0]

# Get all transactions from this IP
single_attack = bin_attacks[bin_attacks['source_ip'] == target_ip].copy()

# Get the BIN prefix from the first card (they should all have the same prefix in a BIN attack)
bin_prefix = single_attack['card_id'].iloc[0][:6]  # De eerste 6 cijfers van het kaart ID zijn de BIN prefix

# Create directed graph for this attack
G = nx.DiGraph()

# Add IP node (root)
G.add_node(target_ip, 
           node_type='ip',
           label=f'IP: {target_ip}',
           layer=0)  # Top layer

# Add card nodes and edges
for card_id in single_attack['card_id'].unique():
    G.add_node(card_id, 
               node_type='card',
               label=f'{card_id[:6]}-{card_id[-4:]}',  # Gebruik BIN prefix + laatste 4 cijfers van kaart ID
               layer=1)  # Second layer
    G.add_edge(target_ip, card_id)

# Add transaction nodes and edges
for _, tx in single_attack.iterrows():
    tx_id = tx['transaction_id']
    G.add_node(tx_id, 
               node_type='transaction',
               label=f'${tx["amount"]:.2f}',
               layer=2,  # Third layer
               is_chargeback=tx['is_chargeback'])
    G.add_edge(tx['card_id'], tx_id)
    
    # Add merchant node and edge if not exists
    if tx['merchant_id'] not in G:
        G.add_node(tx['merchant_id'], 
                  node_type='merchant',
                  label=f'Merchant: {tx["merchant_name"]}',
                  layer=3)  # Bottom layer
    G.add_edge(tx_id, tx['merchant_id'])

# Create the visualization
plt.figure(figsize=(20, 20))

# Create circular layout for each layer
pos = {}
layers = {0: [], 1: [], 2: [], 3: []}

# Group nodes by layer
for node, attr in G.nodes(data=True):
    layers[attr['layer']].append(node)

# Position nodes in circular layout per layer
for layer_num, nodes in layers.items():
    num_nodes = len(nodes)
    if layer_num == 0:  # IP node in center
        pos[nodes[0]] = (0, 0)
    elif layer_num == 3:  # Merchants at bottom
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            pos[node] = (1.5 * np.cos(angle), -2)  # Fixed y position for merchants
    else:  # Cards and transactions in circles
        radius = layer_num * 1.0  # Increasing radius for each layer
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / num_nodes
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

# Define node colors and sizes
node_colors = {
    'ip': '#FF6B6B',      # Red for IP
    'card': '#4ECDC4',    # Turquoise for cards
    'transaction': '#45B7D1',  # Light blue for transactions
    'merchant': '#96CEB4'  # Green for merchants
}

# Draw nodes by type with different sizes
node_sizes = {
    'ip': 3000,
    'card': 1000,
    'transaction': 500,
    'merchant': 2000
}

# Create legend labels
legend_elements = [
    # Nodes
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors['ip'],
               markersize=15, label='IP Address (Fraudster)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors['card'],
               markersize=12, label=f'Card with BIN prefix {bin_prefix}'),  # Gebruik de echte BIN prefix
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors['transaction'],
               markersize=10, label='Transaction (€1-€3)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_colors['merchant'],
               markersize=13, label='Target Merchant'),
    # Edges
    plt.Line2D([0], [0], color='gray', linestyle='-',
               label='Original Transaction'),
    plt.Line2D([0], [0], color='red', linestyle='--',
               label='Chargeback Transaction')
]

# Draw nodes by type
for node_type in ['ip', 'card', 'transaction', 'merchant']:
    nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == node_type]
    nx.draw_networkx_nodes(G, pos,
                          nodelist=nodes,
                          node_color=node_colors[node_type],
                          node_size=node_sizes[node_type])

# Draw edges with different colors and styles for chargebacks
edge_colors = []
edge_styles = []
for u, v in G.edges():
    if G.nodes[u]['node_type'] == 'transaction' and G.nodes[u].get('is_chargeback', False):
        edge_colors.append('red')
        edge_styles.append('dashed')
    else:
        edge_colors.append('gray')
        edge_styles.append('solid')

# Draw all edges
edges = list(G.edges())
for i, (u, v) in enumerate(edges):
    nx.draw_networkx_edges(G, pos,
                          edgelist=[(u, v)],
                          edge_color=edge_colors[i],
                          style=edge_styles[i],
                          arrows=True,
                          arrowsize=10,
                          width=0.5)

# Add labels with smaller font size and slight offset
labels = {node: G.nodes[node]['label'] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title(f'BIN Attack Pattern\nSource IP: {target_ip}', pad=20, fontsize=14)

# Add the custom legend with better positioning and larger font
plt.legend(handles=legend_elements, 
          loc='center left', 
          bbox_to_anchor=(1.1, 0.5),
          fontsize=12,
          title='Pattern Elements',
          title_fontsize=13,
          frameon=True,
          edgecolor='black',
          facecolor='white',
          shadow=True)

# Add pattern statistics in a text box
stats_text = f"Pattern Statistics:\n" \
            f"• Cards Used: {len(single_attack['card_id'].unique())}\n" \
            f"• Total Transactions: {len(single_attack)}\n" \
            f"• Chargebacks: {len(single_attack[single_attack['is_chargeback']])}\n" \
            f"• Time Window: 30 minutes\n" \
            f"• Avg. Amount: €{single_attack['amount'].mean():.2f}"

plt.text(1.1, 0.8, stats_text,
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top')

plt.axis('off')
plt.tight_layout()
plt.savefig('single_bin_attack_tree.png', bbox_inches='tight', dpi=300, facecolor='white')

# Print statistics about this attack
print(f"\nBIN Attack Analysis from IP {target_ip}:")
print(f"Number of cards used: {len(single_attack['card_id'].unique())}")
print(f"Number of transactions: {len(single_attack)}")
print(f"Number of chargebacks: {len(single_attack[single_attack['is_chargeback']])}")
print(f"Number of unique merchants: {len(single_attack['merchant_id'].unique())}")
print(f"Total amount spent: ${single_attack['amount'].sum():.2f}")
print(f"Average transaction amount: ${single_attack['amount'].mean():.2f}")

# Print layer statistics
print("\nLayer Statistics:")
print(f"Layer 0 (IP): {len(layers[0])} node")
print(f"Layer 1 (Cards): {len(layers[1])} nodes")
print(f"Layer 2 (Transactions): {len(layers[2])} nodes")
print(f"Layer 3 (Merchants): {len(layers[3])} nodes") 