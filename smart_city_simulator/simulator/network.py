from typing import Dict, List, Tuple, Optional
import random
import math
import networkx as nx

from models.node import Node, NodeType

class Network:
    """
    Represents the network infrastructure in the smart city simulator.
    Manages connections between nodes and simulates network conditions.
    """
    def __init__(self):
        self.nodes = {}  # node_id -> Node
        self.connections = {}  # (node_id_1, node_id_2) -> {"bandwidth": X, "latency": Y}
        self.graph = nx.Graph()  # For path finding and network analysis
        self.base_failure_probability = 0.001  # Base failure probability per second
        self.congestion_level = {}  # node_id -> congestion level (0.0 - 1.0)
        
    def add_node(self, node: Node) -> None:
        """Add a node to the network"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, type=node.node_type.value)
        self.congestion_level[node.id] = 0.0
        
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the network"""
        if node_id in self.nodes:
            # Remove all connections involving this node
            connections_to_remove = []
            for connection in self.connections:
                if node_id in connection:
                    connections_to_remove.append(connection)
            
            for connection in connections_to_remove:
                del self.connections[connection]
            
            # Remove from graph
            self.graph.remove_node(node_id)
            
            # Remove from node dictionary
            del self.nodes[node_id]
            del self.congestion_level[node_id]
            
            return True
        return False
    
    def connect(self, node1_id: str, node2_id: str, bandwidth: float, latency: float) -> bool:
        """
        Create a connection between two nodes
        
        Args:
            node1_id: ID of first node
            node2_id: ID of second node
            bandwidth: Connection bandwidth in Mbps
            latency: Connection latency in ms
        """
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return False
            
        # Store connection details (ensure connection is always identified by sorted node IDs)
        connection_key = tuple(sorted([node1_id, node2_id]))
        self.connections[connection_key] = {
            "bandwidth": bandwidth,
            "latency": latency,
            "failure_probability": self.base_failure_probability,
            "is_active": True
        }
        
        # Update the graph
        self.graph.add_edge(node1_id, node2_id, bandwidth=bandwidth, 
                           latency=latency, weight=latency)
        
        return True
    
    def disconnect(self, node1_id: str, node2_id: str) -> bool:
        """Remove a connection between two nodes"""
        connection_key = tuple(sorted([node1_id, node2_id]))
        if connection_key in self.connections:
            del self.connections[connection_key]
            self.graph.remove_edge(node1_id, node2_id)
            return True
        return False
    
    def get_connection(self, node1_id: str, node2_id: str) -> Optional[Dict]:
        """Get connection details between two nodes"""
        connection_key = tuple(sorted([node1_id, node2_id]))
        return self.connections.get(connection_key)
    
    def set_connection_status(self, node1_id: str, node2_id: str, is_active: bool) -> bool:
        """Set the active status of a connection (for simulating failures)"""
        connection_key = tuple(sorted([node1_id, node2_id]))
        if connection_key in self.connections:
            self.connections[connection_key]["is_active"] = is_active
            
            # Update graph for path calculations
            if is_active:
                self.graph.add_edge(node1_id, node2_id,
                                  bandwidth=self.connections[connection_key]["bandwidth"],
                                  latency=self.connections[connection_key]["latency"],
                                  weight=self.connections[connection_key]["latency"])
            else:
                if self.graph.has_edge(node1_id, node2_id):
                    self.graph.remove_edge(node1_id, node2_id)
                    
            return True
        return False
    
    def find_shortest_path(self, source_id: str, target_id: str, weight='latency') -> List[str]:
        """Find the shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def calculate_transfer_time(self, source_id: str, target_id: str, data_size_bytes: int) -> float:
        """
        Calculate the time needed to transfer data between nodes
        
        Args:
            source_id: ID of the source node
            target_id: ID of the destination node
            data_size_bytes: Size of data to transfer in bytes
            
        Returns:
            Time in seconds needed for the transfer
        """
        # Find the path
        path = self.find_shortest_path(source_id, target_id)
        if not path or len(path) < 2:
            return float('inf')  # No path found
        
        # Calculate total latency and minimum bandwidth along the path
        total_latency = 0.0  # ms
        min_bandwidth = float('inf')  # Mbps
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i+1]
            connection = self.get_connection(node1, node2)
            
            if not connection or not connection["is_active"]:
                return float('inf')  # Connection is down
                
            latency = connection["latency"]  # ms
            bandwidth = connection["bandwidth"]  # Mbps
            
            # Factor in congestion
            congestion_factor = max(self.congestion_level[node1], self.congestion_level[node2])
            effective_bandwidth = bandwidth * (1 - congestion_factor)
            
            total_latency += latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
        
        # Calculate transfer time
        data_size_mbits = data_size_bytes * 8 / 1_000_000  # Convert bytes to megabits
        transfer_time = data_size_mbits / min_bandwidth  # seconds
        
        # Add latency (convert from ms to s)
        total_time = transfer_time + (total_latency / 1000.0)
        
        return total_time
    
    def update_congestion(self) -> None:
        """Update congestion levels based on node connections and data flow"""
        for node_id in self.nodes:
            # Calculate node's degree (number of connections)
            degree = self.graph.degree(node_id)
            
            # For now, use a simple random model for congestion
            # In a real implementation, this would be based on actual data flow
            if degree > 0:
                # Higher degree = higher chance of congestion
                base_congestion = random.random() * 0.2  # Base random congestion up to 20%
                degree_factor = min(0.8, degree / 10)  # Scale with degree, max 80%
                
                # Node type influences congestion (edge nodes more likely to be congested)
                node = self.nodes[node_id]
                type_factor = {
                    NodeType.EDGE: 0.8,
                    NodeType.FOG: 0.5,
                    NodeType.CLOUD: 0.2
                }.get(node.node_type, 0.5)
                
                # Calculate final congestion level (0.0 - 1.0)
                self.congestion_level[node_id] = min(1.0, base_congestion * degree_factor * type_factor)
            else:
                self.congestion_level[node_id] = 0.0
    
    def simulate_failures(self) -> List[Tuple[str, str]]:
        """
        Simulate random network failures
        
        Returns:
            List of failed connections as (node1_id, node2_id) pairs
        """
        failed_connections = []
        
        for connection_key, connection in self.connections.items():
            if connection["is_active"]:
                # Check if connection fails
                if random.random() < connection["failure_probability"]:
                    node1_id, node2_id = connection_key
                    self.set_connection_status(node1_id, node2_id, False)
                    failed_connections.append(connection_key)
        
        return failed_connections
    
    def recover_random_failures(self, recovery_probability: float = 0.1) -> List[Tuple[str, str]]:
        """
        Recover some failed connections
        
        Args:
            recovery_probability: Probability of recovery for each failed connection
            
        Returns:
            List of recovered connections as (node1_id, node2_id) pairs
        """
        recovered_connections = []
        
        for connection_key, connection in self.connections.items():
            if not connection["is_active"]:
                # Check if connection recovers
                if random.random() < recovery_probability:
                    node1_id, node2_id = connection_key
                    self.set_connection_status(node1_id, node2_id, True)
                    recovered_connections.append(connection_key)
        
        return recovered_connections
    
    def get_all_nodes(self) -> Dict[str, Node]:
        """Get all nodes in the network"""
        return self.nodes.copy()
    
    def get_all_connections(self) -> Dict[Tuple[str, str], Dict]:
        """Get all connections in the network"""
        return self.connections.copy()
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the network"""
        active_connections = sum(1 for conn in self.connections.values() if conn["is_active"])
        
        return {
            "node_count": len(self.nodes),
            "connection_count": len(self.connections),
            "active_connections": active_connections,
            "failed_connections": len(self.connections) - active_connections,
            "average_congestion": sum(self.congestion_level.values()) / max(1, len(self.congestion_level)),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
            "average_shortest_path_length": nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else float('inf')
        }