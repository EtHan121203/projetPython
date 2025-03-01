import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional

from models.node import Node, NodeType
from simulator.network import Network

class NetworkMapVisualization:
    """
    Visualization component for rendering the smart city network.
    Shows nodes and connections with their status.
    """
    def __init__(self, network: Network):
        self.network = network
        self.node_positions = {}  # node_id -> (x, y)
        self.node_colors = {
            NodeType.EDGE: 'lightblue',
            NodeType.FOG: 'lightgreen',
            NodeType.CLOUD: 'lightyellow'
        }
        self.node_sizes = {
            NodeType.EDGE: 100,
            NodeType.FOG: 200,
            NodeType.CLOUD: 300
        }
        self.connection_colors = {
            True: 'green',  # Active connection
            False: 'red'    # Failed connection
        }
    
    def update_node_positions(self):
        """
        Update node positions using either stored locations or network layout algorithm
        """
        # First try to use actual geographical coordinates from nodes
        for node_id, node in self.network.nodes.items():
            if hasattr(node, 'location') and node.location:
                self.node_positions[node_id] = node.location
        
        # For nodes without positions, use spring layout algorithm
        missing_nodes = [node_id for node_id in self.network.nodes.keys() 
                         if node_id not in self.node_positions]
        
        if missing_nodes:
            # Create temporary subgraph for positioning
            temp_graph = nx.Graph()
            for node_id in missing_nodes:
                temp_graph.add_node(node_id)
            
            for (node1, node2) in self.network.connections:
                if node1 in missing_nodes and node2 in missing_nodes:
                    temp_graph.add_edge(node1, node2)
            
            if temp_graph.nodes:
                pos = nx.spring_layout(temp_graph)
                self.node_positions.update(pos)
    
    def draw(self, ax):
        """
        Draw the network graph on the given matplotlib axes
        
        Args:
            ax: Matplotlib axes to draw on
        """
        # Update node positions
        self.update_node_positions()
        
        # Clear any previous content
        ax.clear()
        
        # Create a NetworkX graph for visualization
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.network.nodes.items():
            G.add_node(node_id, type=node.node_type)
        
        # Add edges with properties
        for (node1, node2), connection in self.network.connections.items():
            G.add_edge(node1, node2, is_active=connection["is_active"],
                      bandwidth=connection["bandwidth"], latency=connection["latency"])
        
        # Draw nodes by type
        for node_type in NodeType:
            node_list = [node_id for node_id, node in self.network.nodes.items()
                        if node.node_type == node_type]
            if not node_list:
                continue
                
            node_pos = {node_id: self.node_positions[node_id] for node_id in node_list}
            nx.draw_networkx_nodes(
                G, node_pos, nodelist=node_list, 
                node_color=self.node_colors[node_type],
                node_size=self.node_sizes[node_type],
                ax=ax, alpha=0.8, label=node_type.value
            )
        
        # Draw edges by status
        for status in [True, False]:  # Active, then failed
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_active") == status]
            if edges:
                nx.draw_networkx_edges(
                    G, self.node_positions, edgelist=edges, 
                    edge_color=self.connection_colors[status],
                    width=1.5 if status else 1.0,  # Active connections thicker
                    style='solid' if status else 'dashed',  # Failed connections dashed
                    alpha=0.7, ax=ax
                )
        
        # Draw node labels
        labels = {node_id: self.get_node_label(node) for node_id, node in self.network.nodes.items()}
        nx.draw_networkx_labels(G, self.node_positions, labels=labels, font_size=8, ax=ax)
        
        # Draw edge labels (bandwidth/latency)
        edge_labels = {}
        for (node1, node2), connection in self.network.connections.items():
            bandwidth = connection["bandwidth"]
            latency = connection["latency"]
            edge_labels[(node1, node2)] = f"{bandwidth} Mbps\n{latency} ms"
        
        nx.draw_networkx_edge_labels(
            G, self.node_positions, edge_labels=edge_labels,
            font_size=6, ax=ax, rotate=False
        )
        
        # Set plot properties
        ax.set_title("Smart City Network Map")
        ax.legend(loc='upper right')
        ax.set_axis_off()
    
    def get_node_label(self, node: Node) -> str:
        """Create a concise label for a node with key info"""
        # Get number of services running on this node
        service_count = len(getattr(node, 'running_services', {}))
        
        if hasattr(node, 'name'):
            name = node.name
        else:
            name = node.id[:8]  # Truncate ID if no name
        
        # Add resource usage if available
        if hasattr(node, 'resources'):
            cpu_usage, ram_usage, _ = node.resources.usage_percentage()
            return f"{name}\n{service_count} svc\nCPU:{cpu_usage:.0f}%\nRAM:{ram_usage:.0f}%"
        
        return f"{name}\n{service_count} svc"
    
    def highlight_path(self, ax, path: List[str], color='orange', linewidth=3.0):
        """
        Highlight a path between nodes in the network
        
        Args:
            ax: Matplotlib axes to draw on
            path: List of node IDs forming the path
            color: Color to use for highlighting
            linewidth: Width of the highlighting line
        """
        if not path or len(path) < 2:
            return
        
        edges = list(zip(path[:-1], path[1:]))
        
        # Get positions of nodes along the path
        path_pos = {node_id: self.node_positions[node_id] for node_id in path}
        
        # Draw highlighted edges
        nx.draw_networkx_edges(
            nx.DiGraph(), path_pos, edgelist=edges, 
            edge_color=color, width=linewidth,
            arrows=True, ax=ax, connectionstyle='arc3,rad=0.1'  # Curved edges
        )
    
    def draw_congestion(self, ax):
        """
        Draw network congestion levels
        
        Args:
            ax: Matplotlib axes to draw on
        """
        congestion = self.network.congestion_level
        
        # Skip if no congestion data
        if not congestion:
            return
        
        # Draw congestion as node color intensity
        for node_id, level in congestion.items():
            if node_id not in self.node_positions or node_id not in self.network.nodes:
                continue
                
            node = self.network.nodes[node_id]
            base_color = self.node_colors[node.node_type]
            
            # Mix base color with red based on congestion level
            # Higher congestion = more red
            r = min(1.0, 0.5 + level * 0.5)  # Red component increases with congestion
            g = max(0.0, 0.5 - level * 0.5)  # Green component decreases with congestion
            b = max(0.0, 0.5 - level * 0.5)  # Blue component decreases with congestion
            
            congestion_color = (r, g, b)
            
            # Draw circle representing congestion level
            pos = self.node_positions[node_id]
            size = self.node_sizes[node.node_type] * 0.01  # Scale factor
            circle = plt.Circle(pos, size * (0.5 + level), 
                               color=congestion_color, alpha=0.6)
            ax.add_patch(circle)
    
    def save_to_file(self, filename: str):
        """
        Save the current network visualization to a file
        
        Args:
            filename: Path to save the image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        self.draw(ax)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)