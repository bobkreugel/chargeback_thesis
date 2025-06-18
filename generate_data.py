from src.config.config_manager import ConfigurationManager
from src.engine.transaction_engine import TransactionEngine
import os
import logging
import networkx as nx
import pickle
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_graph(graph, filepath):
    """Save a NetworkX graph to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)

def prepare_graph_for_graphml(graph):
    """Prepare graph for GraphML export by converting non-supported types to strings."""
    graph = graph.copy()
    
    # Replace None values and convert datetime objects in node attributes
    for node, attrs in graph.nodes(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
            elif isinstance(value, datetime):
                attrs[key] = value.isoformat()
            elif isinstance(value, type):
                attrs[key] = value.__name__
    
    # Replace None values and convert datetime objects in edge attributes
    for u, v, attrs in graph.edges(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
            elif isinstance(value, datetime):
                attrs[key] = value.isoformat()
            elif isinstance(value, type):
                attrs[key] = value.__name__
    
    return graph

def calculate_statistics(graph):
    """Calculate statistics about the generated dataset."""
    stats = {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'node_types': {},
        'transactions': {
            'total': 0,
            'normal': 0,
            'legitimate_chargebacks': 0,
            'fraudulent': 0,
            'bin_attacks': 0,
            'serial_chargebacks': 0,
            'friendly_fraud': 0,
            'geographic_mismatch': 0
        }
    }
    
    # Count node types
    for node, attr in graph.nodes(data=True):
        node_type = attr.get('node_type', 'unknown')
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count transaction types
        if node_type == 'transaction':
            stats['transactions']['total'] += 1
            
            # Skip chargeback transactions for fraud pattern counting
            if attr.get('is_chargeback', False):
                if not attr.get('is_fraudulent', False):
                    stats['transactions']['legitimate_chargebacks'] += 1
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
                        stats['transactions']['fraudulent'] += 1
                        
                        # Check fraud type
                        fraud_type = customer_attr.get('fraud_type')
                        if fraud_type == 'bin_attack':
                            stats['transactions']['bin_attacks'] += 1
                        elif fraud_type == 'serial_chargeback':
                            stats['transactions']['serial_chargebacks'] += 1
                        elif fraud_type == 'friendly_fraud':
                            stats['transactions']['friendly_fraud'] += 1
                        elif fraud_type == 'geographic_mismatch':
                            stats['transactions']['geographic_mismatch'] += 1
                    else:
                        stats['transactions']['normal'] += 1
    
    return stats

def main():
    # Use fixed output directory
    output_dir = "output/dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configuration (uses default config if no path provided)
    config_manager = ConfigurationManager()
    
    # Save configuration used
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config_manager.get_config(), f, indent=2)
    
    # Initialize transaction engine with no fixed seed for random data each time
    engine = TransactionEngine(config_manager)
    
    # Generate base population
    logger.info("Generating base population...")
    engine.generate_base_population()
    
    # Generate normal transactions with legitimate chargebacks
    logger.info("Generating transactions...")
    engine.generate_normal_transactions()
    
    # Inject fraud patterns
    logger.info("Injecting fraud patterns...")
    engine._inject_patterns()
    
    # Get the complete graph
    graph = engine.get_graph()
    
    # Calculate and save statistics
    logger.info("Calculating dataset statistics...")
    stats = calculate_statistics(graph)
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Log key statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total Transactions: {stats['transactions']['total']}")
    logger.info(f"Normal Transactions: {stats['transactions']['normal']}")
    logger.info(f"Legitimate Chargebacks: {stats['transactions']['legitimate_chargebacks']}")
    logger.info("\nFraud Statistics:")
    logger.info(f"Total Fraudulent: {stats['transactions']['fraudulent']}")
    logger.info(f"BIN Attacks: {stats['transactions']['bin_attacks']} ({stats['transactions']['bin_attacks']/stats['transactions']['fraudulent']*100:.1f}%)")
    logger.info(f"Serial Chargebacks: {stats['transactions']['serial_chargebacks']} ({stats['transactions']['serial_chargebacks']/stats['transactions']['fraudulent']*100:.1f}%)")
    logger.info(f"Friendly Fraud: {stats['transactions']['friendly_fraud']} ({stats['transactions']['friendly_fraud']/stats['transactions']['fraudulent']*100:.1f}%)")
    logger.info(f"Geographic Mismatch: {stats['transactions']['geographic_mismatch']} ({stats['transactions']['geographic_mismatch']/stats['transactions']['fraudulent']*100:.1f}%)")
    
    # Export to CSV
    logger.info("\nExporting data to CSV...")
    engine.export_to_csv(output_dir)
    
    # Export graph formats
    logger.info("Exporting graph formats...")
    save_graph(graph, os.path.join(output_dir, "transaction_graph.gpickle"))
    
    # Prepare and save GraphML
    graphml_graph = prepare_graph_for_graphml(graph)
    nx.write_graphml(graphml_graph, os.path.join(output_dir, "transaction_graph.graphml"))
    
    logger.info(f"\nData generation complete. Files saved in {output_dir}/")
    logger.info("Generated files:")
    logger.info("- transaction_graph.gpickle (Complete graph with all attributes)")
    logger.info("- transaction_graph.graphml (Graph in GraphML format)")
    logger.info("- transactions.csv (All transactions)")
    logger.info("- customers.csv (Customer information)")
    logger.info("- merchants.csv (Merchant information)")
    logger.info("- cards.csv (Card information)")
    logger.info("- config.json (Configuration used)")
    logger.info("- statistics.json (Dataset statistics)")

if __name__ == "__main__":
    main() 