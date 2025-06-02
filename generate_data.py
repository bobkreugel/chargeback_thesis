from src.config.config_manager import ConfigurationManager
from src.engine.transaction_engine import TransactionEngine
import os
import logging
import networkx as nx
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_graph(graph, filepath):
    """Save a NetworkX graph to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)

def prepare_graph_for_graphml(graph):
    """Prepare graph for GraphML export by replacing None values with empty strings."""
    graph = graph.copy()
    
    # Replace None values in node attributes
    for node, attrs in graph.nodes(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
    
    # Replace None values in edge attributes
    for u, v, attrs in graph.edges(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
    
    return graph

def main():
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configuration (uses default config if no path provided)
    config_manager = ConfigurationManager()
    
    # Initialize transaction engine
    engine = TransactionEngine(config_manager)
    
    # Generate base population
    logger.info("Generating base population...")
    engine.generate_base_population()
    
    # Generate normal transactions with legitimate chargebacks
    logger.info("Generating transactions...")
    engine.generate_normal_transactions()
    
    # Export to CSV
    logger.info("Exporting data to CSV...")
    engine.export_to_csv(output_dir)
    
    # Export graph formats
    logger.info("Exporting graph formats...")
    graph = engine.get_graph()
    save_graph(graph, os.path.join(output_dir, "transaction_graph.gpickle"))
    
    # Prepare and save GraphML
    graphml_graph = prepare_graph_for_graphml(graph)
    nx.write_graphml(graphml_graph, os.path.join(output_dir, "transaction_graph.graphml"))
    
    logger.info(f"Data generation complete. Files saved in {output_dir}/")

if __name__ == "__main__":
    main() 