"""
Causal inference framework for TSA prediction project focusing on time-series confounding.
"""
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import json

logger = logging.getLogger(__name__)

class CausalGraph:
    """
    Implements a causal DAG for TSA passenger volume analysis with focus on confounding.
    """
    
    def __init__(self):
        """Initialize the causal graph."""
        self.graph = nx.DiGraph()
        self._setup_graph()
        self._identify_confounders()
        
    def _setup_graph(self):
        """Define the causal relationships in the DAG."""
        # Node definitions with clear roles in the causal structure
        nodes = {
            # Primary factors
            'airline_prices': 'Market airfare prices',
            'weather_severity': 'Weather conditions at major airports',
            'passenger_volume': 'TSA checkpoint volume (target)',
            
            # Potential confounders
            'season': 'Seasonal patterns',
            'day_of_week': 'Day of week effects',
            'holiday': 'Holiday periods',
            'economic_activity': 'General economic conditions',
            
            # Mediating variables
            'business_travel': 'Business travel demand',
            'leisure_travel': 'Leisure travel demand',
        }
        
        # Add nodes with their roles
        for node_id, description in nodes.items():
            self.graph.add_node(node_id, description=description)
        
        # Add edges with clear causal pathways
        edges = [
            # Direct effects on passenger volume
            ('airline_prices', 'passenger_volume'),
            ('weather_severity', 'passenger_volume'),
            ('business_travel', 'passenger_volume'),
            ('leisure_travel', 'passenger_volume'),
            
            # Confounder relationships
            ('season', 'weather_severity'),
            ('season', 'leisure_travel'),
            ('season', 'airline_prices'),
            
            ('holiday', 'leisure_travel'),
            ('holiday', 'airline_prices'),
            
            ('economic_activity', 'business_travel'),
            ('economic_activity', 'leisure_travel'),
            
            ('day_of_week', 'business_travel'),
            ('day_of_week', 'passenger_volume')
        ]
        
        self.graph.add_edges_from(edges)
    
    def _identify_confounders(self):
        """
        Identify confounders for each causal relationship.
        Uses backdoor criterion to find minimal adjustment sets.
        """
        self.confounders = {}
        
        # For each potential cause-effect pair
        for cause in self.graph.nodes():
            for effect in self.graph.nodes():
                if cause != effect and nx.has_path(self.graph, cause, effect):
                    # Find all backdoor paths
                    backdoor_paths = self._find_backdoor_paths(cause, effect)
                    
                    # Identify minimal adjustment set
                    adjustment_set = self._get_minimal_adjustment_set(
                        cause, effect, backdoor_paths)
                    
                    if adjustment_set:
                        self.confounders[(cause, effect)] = adjustment_set
    
    def _find_backdoor_paths(self, cause: str, effect: str) -> List[List[str]]:
        """
        Find all backdoor paths between cause and effect.
        
        Args:
            cause: Causal variable
            effect: Effect variable
            
        Returns:
            List of paths that could create confounding
        """
        # Remove direct edge from cause to effect if it exists
        if self.graph.has_edge(cause, effect):
            self.graph.remove_edge(cause, effect)
            
        # Find all paths from parents of cause to effect
        backdoor_paths = []
        for parent in self.graph.predecessors(cause):
            for path in nx.all_simple_paths(self.graph, parent, effect):
                if cause not in path:  # Ensure it's a backdoor path
                    backdoor_paths.append(path)
                    
        # Restore direct edge
        self.graph.add_edge(cause, effect)
        
        return backdoor_paths
    
    def _get_minimal_adjustment_set(self, cause: str, effect: str, 
                                  backdoor_paths: List[List[str]]) -> Set[str]:
        """
        Get minimal set of variables needed to block backdoor paths.
        
        Args:
            cause: Causal variable
            effect: Effect variable
            backdoor_paths: List of backdoor paths
            
        Returns:
            Set of variables that blocks all backdoor paths
        """
        if not backdoor_paths:
            return set()
            
        # Get all variables in backdoor paths
        variables = set()
        for path in backdoor_paths:
            variables.update(path)
            
        # Remove cause and effect
        variables.discard(cause)
        variables.discard(effect)
        
        return variables
    
    def estimate_causal_effect(self, data: pd.DataFrame,
                             cause: str,
                             effect: str = 'passenger_volume',
                             time_controls: bool = True) -> Dict:
        """
        Estimate causal effect controlling for appropriate confounders.
        
        Args:
            data: DataFrame containing variables
            cause: Causal variable of interest
            effect: Effect variable (default: passenger_volume)
            time_controls: Whether to include time-based controls
            
        Returns:
            Dict with effect estimate and diagnostics
        """
        # Get confounders for this relationship
        required_confounders = self.confounders.get((cause, effect), set())
        
        # Add time-based controls if requested
        if time_controls:
            time_controls = ['season', 'day_of_week', 'holiday']
            required_confounders.update(c for c in time_controls 
                                     if c in data.columns)
        
        # Prepare model data
        available_confounders = [c for c in required_confounders 
                               if c in data.columns]
        
        # Create design matrix with confounders
        X = pd.DataFrame(index=data.index)
        X[cause] = data[cause]
        
        for confounder in available_confounders:
            if pd.api.types.is_categorical_dtype(data[confounder]):
                # Create dummies for categorical confounders
                dummies = pd.get_dummies(data[confounder], prefix=confounder)
                X = pd.concat([X, dummies], axis=1)
            else:
                X[confounder] = data[confounder]
        
        # Add constant
        X = sm.add_constant(X)
        y = data[effect]
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Calculate confidence intervals
        conf_int = results.conf_int().loc[cause]
        
        return {
            'effect': results.params[cause],
            'std_err': results.bse[cause],
            'p_value': results.pvalues[cause],
            'conf_int_lower': conf_int[0],
            'conf_int_upper': conf_int[1],
            'r_squared': results.rsquared,
            'n_obs': results.nobs,
            'controlled_confounders': available_confounders,
            'missing_confounders': required_confounders - set(available_confounders)
        }
    
    def plot(self, save_path: Optional[Path] = None):
        """Plot the causal graph with confounders highlighted."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw regular nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                             node_size=2000, alpha=0.7)
        
        # Draw confounder nodes
        confounder_nodes = set()
        for confounders in self.confounders.values():
            confounder_nodes.update(confounders)
        
        nx.draw_networkx_nodes(self.graph, pos, 
                             nodelist=list(confounder_nodes),
                             node_color='lightgreen',
                             node_size=2000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray',
                             arrows=True, arrowsize=20)
        
        # Add labels
        labels = nx.get_node_attributes(self.graph, 'description')
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Causal Graph for TSA Passenger Volume\n(Confounders in Green)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Example usage
    causal_graph = CausalGraph()
    
    # Print confounders for key relationships
    print("\nConfounders for key relationships:")
    key_causes = ['weather_severity', 'airline_prices', 'holiday']
    for cause in key_causes:
        confounders = causal_graph.confounders.get((cause, 'passenger_volume'), set())
        print(f"\n{cause} -> passenger_volume")
        print(f"Required confounders: {confounders}")