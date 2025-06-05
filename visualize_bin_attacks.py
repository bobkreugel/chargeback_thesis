import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import numpy as np

def load_config():
    """Load the configuration file to get the configured BIN prefixes."""
    config_path = Path('src/config/default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_graph(filepath):
    """Load the NetworkX graph from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_card_patterns(graph, config):
    """
    Analyze transaction patterns for BIN attack cards vs normal cards.
    Returns statistics for both groups.
    """
    bin_prefixes = config['fraud_patterns']['bin_attack']['bin_prefixes']
    bin_window = config['fraud_patterns']['bin_attack']['time_window']['minutes']
    
    # Initialize data structures
    bin_cards = defaultdict(list)  # BIN prefix -> list of cards
    normal_cards = []
    card_transactions = defaultdict(list)  # card -> list of transactions
    
    # Collect cards and their transactions
    for node, attrs in graph.nodes(data=True):
        if attrs.get('node_type') == 'card':
            card_id = str(node)
            # Check if it's a BIN attack card
            is_bin_card = False
            for prefix in bin_prefixes:
                if card_id.startswith(prefix):
                    bin_cards[prefix].append(card_id)
                    is_bin_card = True
                    break
            if not is_bin_card and not attrs.get('is_fraudulent', False):
                normal_cards.append(card_id)
                
            # Get all transactions for this card
            for _, tx in graph.out_edges(node):
                tx_attrs = graph.nodes[tx]
                if tx_attrs.get('node_type') == 'transaction':
                    # Find the merchant for this transaction
                    merchant = None
                    for m, t in graph.in_edges(tx):
                        if graph.nodes[m].get('node_type') == 'merchant':
                            merchant = m
                            break
                    if merchant:
                        tx_attrs['merchant_id'] = merchant
                    card_transactions[card_id].append(tx_attrs)

    # Analyze patterns
    stats = {
        'bin_cards': {
            'total_cards': sum(len(cards) for cards in bin_cards.values()),
            'avg_txs_per_card': 0,
            'avg_amount': 0,
            'chargeback_rate': 0,
            'merchant_reuse': 0
        },
        'normal_cards': {
            'total_cards': len(normal_cards),
            'avg_txs_per_card': 0,
            'avg_amount': 0,
            'chargeback_rate': 0,
            'merchant_reuse': 0
        }
    }
    
    # Analyze BIN attack cards
    for prefix, cards in bin_cards.items():
        for card in cards:
            txs = card_transactions[card]
            if not txs:
                continue
            
            # Filter out chargeback transactions for counting
            original_txs = [tx for tx in txs if not tx.get('is_chargeback', False)]
            chargeback_txs = [tx for tx in txs if tx.get('is_chargeback', False)]
            
            if not original_txs:
                continue
                
            # Calculate metrics
            amounts = [tx['amount'] for tx in original_txs]
            merchants = set(tx.get('merchant_id') for tx in original_txs if 'merchant_id' in tx)
            
            stats['bin_cards']['avg_txs_per_card'] += len(original_txs)
            stats['bin_cards']['avg_amount'] += sum(amounts)
            stats['bin_cards']['chargeback_rate'] += len(chargeback_txs) / len(original_txs)
            stats['bin_cards']['merchant_reuse'] += (len(original_txs) - len(merchants)) / len(original_txs) if len(original_txs) > 0 else 0
    
    # Analyze normal cards
    for card in normal_cards:
        txs = card_transactions[card]
        if not txs:
            continue
            
        # Filter out chargeback transactions for counting
        original_txs = [tx for tx in txs if not tx.get('is_chargeback', False)]
        chargeback_txs = [tx for tx in txs if tx.get('is_chargeback', False)]
        
        if not original_txs:
            continue
            
        # Calculate metrics
        amounts = [tx['amount'] for tx in original_txs]
        merchants = set(tx.get('merchant_id') for tx in original_txs if 'merchant_id' in tx)
        
        stats['normal_cards']['avg_txs_per_card'] += len(original_txs)
        stats['normal_cards']['avg_amount'] += sum(amounts)
        stats['normal_cards']['chargeback_rate'] += len(chargeback_txs) / len(original_txs)
        stats['normal_cards']['merchant_reuse'] += (len(original_txs) - len(merchants)) / len(original_txs) if len(original_txs) > 0 else 0
    
    # Calculate averages
    for card_type in ['bin_cards', 'normal_cards']:
        total_cards = stats[card_type]['total_cards']
        if total_cards > 0:
            stats[card_type]['avg_txs_per_card'] /= total_cards
            stats[card_type]['avg_amount'] /= total_cards
            stats[card_type]['chargeback_rate'] /= total_cards
            stats[card_type]['merchant_reuse'] /= total_cards
    
    return stats, bin_cards

def plot_comparison(stats, config):
    """Create comparison plots between BIN attack cards and normal cards."""
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('BIN Attack Cards vs Normal Cards Comparison', fontsize=16, y=1.05)
    
    # Common labels
    labels = ['BIN Attack\nCards', 'Normal\nCards']
    
    # Plot 1: Average transactions per card
    tx_counts = [stats['bin_cards']['avg_txs_per_card'], stats['normal_cards']['avg_txs_per_card']]
    bars1 = ax1.bar(labels, tx_counts)
    ax1.set_title('Average Transactions per Card')
    ax1.grid(True, alpha=0.3)
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Plot 2: Average amount per card (log scale)
    amounts = [stats['bin_cards']['avg_amount'], stats['normal_cards']['avg_amount']]
    bars2 = ax2.bar(labels, amounts)
    ax2.set_title('Average Total Amount per Card\n(log scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'€{height:.2f}',
                ha='center', va='bottom')
    
    # Plot 3: Chargeback rate
    cb_rates = [stats['bin_cards']['chargeback_rate'] * 100, stats['normal_cards']['chargeback_rate'] * 100]
    bars3 = ax3.bar(labels, cb_rates)
    ax3.set_title('Chargeback Rate')
    ax3.set_ylabel('Percentage')
    ax3.grid(True, alpha=0.3)
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('bin_attack_validation.png', bbox_inches='tight', dpi=300)
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
    
    # Analyze patterns
    print("Analyzing transaction patterns...")
    stats, bin_cards = analyze_card_patterns(graph, config)
    
    # Print statistics
    print("\nBIN Attack Pattern Validation Statistics:")
    print("\nBIN Attack Cards:")
    print(f"- Number of cards: {stats['bin_cards']['total_cards']}")
    print(f"- Average transactions per card: {stats['bin_cards']['avg_txs_per_card']:.2f}")
    print(f"- Average total amount per card: €{stats['bin_cards']['avg_amount']:.2f}")
    print(f"- Chargeback rate: {stats['bin_cards']['chargeback_rate']*100:.1f}%")
    print(f"- Merchant reuse ratio: {stats['bin_cards']['merchant_reuse']:.2f}")
    
    print("\nNormal Cards:")
    print(f"- Number of cards: {stats['normal_cards']['total_cards']}")
    print(f"- Average transactions per card: {stats['normal_cards']['avg_txs_per_card']:.2f}")
    print(f"- Average total amount per card: €{stats['normal_cards']['avg_amount']:.2f}")
    print(f"- Chargeback rate: {stats['normal_cards']['chargeback_rate']*100:.1f}%")
    print(f"- Merchant reuse ratio: {stats['normal_cards']['merchant_reuse']:.2f}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_comparison(stats, config)
    print("Validation plots saved as 'bin_attack_validation.png'")

if __name__ == '__main__':
    main() 