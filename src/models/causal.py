"""
Causal inference framework for TSA prediction project.
Uses DAGs to understand relationships between variables affecting TSA check-in numbers.
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
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CausalGraph:
    """
    Implements a causal DAG for TSA passenger volume analysis.
    Provides insights for trading decisions.
    """
    
    def __init__(self):
        """Initialize the causal graph."""
        self.graph = nx.DiGraph()
        self._setup_graph()
        self._identify_confounders()
        self.causal_effects = {}
        
    def _setup_graph(self):
        """Define the causal relationships in the DAG."""
        # Core nodes (primary factors)
        nodes = {
            'tsa_volume': {
                'name': 'TSA Passenger Volume',
                'type': 'target',
                'description': 'Weekly TSA checkpoint volume'
            },
            'airline_prices': {
                'name': 'Airline Ticket Prices',
                'type': 'price',
                'description': 'Average airline ticket prices'
            },
            'weather_severity': {
                'name': 'Weather Severity',
                'type': 'external',
                'description': 'Aggregate weather conditions at major airports'
            },
            
            # Seasonal and calendar factors
            'season': {
                'name': 'Seasonal Effects',
                'type': 'temporal',
                'description': 'Seasonal travel patterns'
            },
            'day_of_week': {
                'name': 'Day of Week',
                'type': 'temporal',
                'description': 'Day of week effects'
            },
            'holidays': {
                'name': 'Holidays',
                'type': 'temporal',
                'description': 'Major holidays and events'
            },
            
            # Economic factors
            'gdp': {
                'name': 'GDP',
                'type': 'economic',
                'description': 'Gross Domestic Product'
            },
            'employment': {
                'name': 'Employment',
                'type': 'economic',
                'description': 'Employment levels'
            },
            'consumer_confidence': {
                'name': 'Consumer Confidence',
                'type': 'economic',
                'description': 'Consumer confidence index'
            },
            
            # Travel specific
            'hotel_prices': {
                'name': 'Hotel Prices',
                'type': 'price',
                'description': 'Average hotel prices'
            },
            'business_travel': {
                'name': 'Business Travel',
                'type': 'travel',
                'description': 'Business travel demand'
            },
            'leisure_travel': {
                'name': 'Leisure Travel',
                'type': 'travel',
                'description': 'Leisure travel demand'
            }
        }
        
        # Add nodes with attributes
        for node_id, attrs in nodes.items():
            self.graph.add_node(node_id, **attrs)
        
        # Define edges with causal relationships
        edges = [
            # Direct effects on TSA volume
            ('airline_prices', 'tsa_volume', {'strength': 'strong', 'lag': 1}),
            ('weather_severity', 'tsa_volume', {'strength': 'medium', 'lag': 0}),
            ('business_travel', 'tsa_volume', {'strength': 'strong', 'lag': 1}),
            ('leisure_travel', 'tsa_volume', {'strength': 'strong', 'lag': 1}),
            
            # Seasonal and calendar effects
            ('season', 'leisure_travel', {'strength': 'strong', 'lag': 0}),
            ('season', 'airline_prices', {'strength': 'medium', 'lag': 1}),
            ('holidays', 'leisure_travel', {'strength': 'strong', 'lag': 2}),
            ('holidays', 'airline_prices', {'strength': 'strong', 'lag': 2}),
            ('day_of_week', 'tsa_volume', {'strength': 'strong', 'lag': 0}),
            
            # Economic effects
            ('gdp', 'business_travel', {'strength': 'medium', 'lag': 3}),
            ('gdp', 'leisure_travel', {'strength': 'medium', 'lag': 3}),
            ('employment', 'business_travel', {'strength': 'strong', 'lag': 2}),
            ('consumer_confidence', 'leisure_travel', {'strength': 'medium', 'lag': 2}),
            
            # Price interactions
            ('hotel_prices', 'leisure_travel', {'strength': 'medium', 'lag': 1}),
            ('airline_prices', 'leisure_travel', {'strength': 'strong', 'lag': 1})
        ]
        
        self.graph.add_edges_from(edges)
        
    def _identify_confounders(self):
        """Identify confounders for each causal relationship."""
        self.confounders = {}
        for cause in self.graph.nodes():
            for effect in self.graph.nodes():
                if cause != effect and nx.has_path(self.graph, cause, effect):
                    backdoor_paths = self._find_backdoor_paths(cause, effect)
                    adjustment_set = self._get_minimal_adjustment_set(
                        cause, effect, backdoor_paths
                    )
                    if adjustment_set:
                        self.confounders[(cause, effect)] = adjustment_set
        
    def _find_backdoor_paths(self, cause: str, effect: str) -> List[List[str]]:
        """Find all backdoor paths between cause and effect."""
        if self.graph.has_edge(cause, effect):
            self.graph.remove_edge(cause, effect)
            
        backdoor_paths = []
        for parent in self.graph.predecessors(cause):
            for path in nx.all_simple_paths(self.graph, parent, effect):
                if cause not in path:
                    backdoor_paths.append(path)
                    
        if self.graph.has_edge(cause, effect) == False:
            self.graph.add_edge(cause, effect)
            
        return backdoor_paths
    
    def _get_minimal_adjustment_set(self, cause: str, effect: str,
                                  backdoor_paths: List[List[str]]) -> Set[str]:
        """Get minimal set of variables needed to block backdoor paths."""
        if not backdoor_paths:
            return set()
            
        variables = set()
        for path in backdoor_paths:
            variables.update(path)
            
        variables.discard(cause)
        variables.discard(effect)
        
        return variables
    
    def estimate_causal_effect(self, 
                             data: pd.DataFrame,
                             cause: str,
                             effect: str = 'tsa_volume',
                             control_confounding: bool = True) -> Dict:
        """
        Estimate causal effect controlling for confounders.
        
        Args:
            data: DataFrame containing variables
            cause: Causal variable of interest
            effect: Effect variable
            control_confounding: Whether to control for confounders
            
        Returns:
            Dict with effect estimates and diagnostics
        """
        try:
            # Get confounders if requested
            confounders = []
            if control_confounding:
                confounders = list(
                    self.confounders.get((cause, effect), set())
                )
                confounders = [c for c in confounders if c in data.columns]
            
            # Prepare model data
            X = pd.DataFrame(index=data.index)
            X[cause] = data[cause]
            
            # Add confounders
            for confounder in confounders:
                if pd.api.types.is_categorical_dtype(data[confounder]):
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
            
            effect_estimate = {
                'effect': results.params[cause],
                'std_err': results.bse[cause],
                'p_value': results.pvalues[cause],
                'conf_int_lower': conf_int[0],
                'conf_int_upper': conf_int[1],
                'r_squared': results.rsquared,
                'n_obs': results.nobs,
                'controlled_confounders': confounders
            }
            
            # Store effect for later use
            self.causal_effects[(cause, effect)] = effect_estimate
            
            return effect_estimate
            
        except Exception as e:
            logger.error(f"Error estimating causal effect: {str(e)}")
            raise
    
    def get_trading_factors(self) -> Dict[str, float]:
        """
        Get causal factors relevant for trading decisions.
        
        Returns:
            Dict mapping factors to their causal strength
        """
        trading_factors = {}
        
        # Get direct effects on TSA volume
        for pred in self.graph.predecessors('tsa_volume'):
            edge_data = self.graph.edges[pred, 'tsa_volume']
            
            # Get stored causal effect if available
            effect_size = 0.0
            if (pred, 'tsa_volume') in self.causal_effects:
                effect = self.causal_effects[(pred, 'tsa_volume')]
                effect_size = abs(effect['effect'])
            
            trading_factors[pred] = {
                'strength': edge_data['strength'],
                'lag': edge_data['lag'],
                'effect_size': effect_size
            }
            
        return trading_factors
    
    def get_risk_factors(self) -> List[Dict]:
        """
        Identify high-risk causal factors.
        
        Returns:
            List of risk factors with their properties
        """
        risk_factors = []
        
        for node, attrs in self.graph.nodes(data=True):
            # Skip target variable
            if attrs.get('type') == 'target':
                continue
                
            # Check number of downstream effects
            n_effects = len(list(nx.descendants(self.graph, node)))
            
            # Get stored causal effect if available
            effect_size = 0.0
            if (node, 'tsa_volume') in self.causal_effects:
                effect = self.causal_effects[(node, 'tsa_volume')]
                effect_size = abs(effect['effect'])
            
            risk_factors.append({
                'factor': node,
                'type': attrs.get('type'),
                'n_effects': n_effects,
                'effect_size': effect_size
            })
        
        # Sort by effect size
        risk_factors.sort(key=lambda x: x['effect_size'], reverse=True)
        return risk_factors
    
    def plot(self, save_path: Optional[Path] = None):
        """Plot the causal graph with factor types highlighted."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Color nodes by type
        colors = {
            'target': 'lightblue',
            'price': 'lightgreen',
            'external': 'lightgray',
            'temporal': 'lightpink',
            'economic': 'lightyellow',
            'travel': 'lightcoral'
        }
        
        for node_type, color in colors.items():
            nodes = [n for n, attr in self.graph.nodes(data=True) 
                    if attr.get('type') == node_type]
            nx.draw_networkx_nodes(self.graph, pos, 
                                 nodelist=nodes,
                                 node_color=color,
                                 node_size=2000, alpha=0.7)
        
        # Draw edges with arrows indicating strength
        edge_colors = {'strong': 'black', 'medium': 'gray', 'weak': 'lightgray'}
        
        for strength, color in edge_colors.items():
            edges = [(u, v) for (u, v, d) in self.graph.edges(data=True)
                    if d.get('strength') == strength]
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=edges,
                                 edge_color=color,
                                 arrows=True,
                                 arrowsize=20)
        
        # Add labels
        labels = nx.get_node_attributes(self.graph, 'name')
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Causal Graph for TSA Passenger Volume")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save(self, path: Path):
        """Save the causal graph structure."""
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load(self, path: Path):
        """Load a causal graph structure."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        self._identify_confounders()

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    data = pd.DataFrame({
        'airline_prices': np.random.normal(100, 20, n_samples),
        'weather_severity': np.random.normal(0, 1, n_samples),
        'season': (np.arange(n_samples) % 4),  # 4 seasons
        'gdp': np.random.normal(50000, 1000, n_samples),
        'employment': np.random.normal(95, 2, n_samples),
        'tsa_volume': np.random.normal(2000000, 200000, n_samples)
    })
    
    # Create causal graph
    causal_graph = CausalGraph()
    
    # Estimate some causal effects
    effect = causal_graph.estimate_causal_effect(
        data=data,
        cause='airline_prices',
        effect='tsa_volume'
    )
    print("\nCausal effect of airline prices:")
    print(effect)
    
    # Get trading factors
    trading_factors = causal_graph.get_trading_factors()
    print("\nTrading factors:")
    for factor, info in trading_factors.items():
        print(f"{factor}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Get risk factors
    risk_factors = causal_graph.get_risk_factors()
    print("\nRisk factors (top 3):")
    for factor in risk_factors[:3]:
        print(f"{factor['factor']}:")
        print(f"  Type: {factor['type']}")
        print(f"  Number of effects: {factor['n_effects']}")
        print(f"  Effect size: {factor['effect_size']:.3f}")
    
    # Plot the graph
    causal_graph.plot(Path("causal_graph.png"))
    print("\nCausal graph saved to causal_graph.png")
    
    # Example of saving and loading
    causal_graph.save(Path("causal_graph.json"))
    new_graph = CausalGraph()
    new_graph.load(Path("causal_graph.json"))
    
    print("\nConfounders for airline_prices -> tsa_volume:")
    print(new_graph.confounders.get(('airline_prices', 'tsa_volume'), set()))