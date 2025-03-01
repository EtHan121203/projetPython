# models/node.py
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class NodeType(Enum):
    EDGE = "edge"
    FOG = "fog"
    CLOUD = "cloud"

class NodeState(Enum):
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
    OVERLOADED = "overloaded"

@dataclass
class ComputeResources:
    cpu_cores: int
    cpu_frequency: float  # GHz
    ram: int  # MB
    storage: int  # MB
    available_cpu: int = None
    available_ram: int = None
    available_storage: int = None
    
    def __post_init__(self):
        if self.available_cpu is None:
            self.available_cpu = self.cpu_cores
        if self.available_ram is None:
            self.available_ram = self.ram
        if self.available_storage is None:
            self.available_storage = self.storage
    
    def allocate(self, cpu: int, ram: int, storage: int) -> bool:
        """Allocates resources if available"""
        if (self.available_cpu >= cpu and 
            self.available_ram >= ram and 
            self.available_storage >= storage):
            self.available_cpu -= cpu
            self.available_ram -= ram
            self.available_storage -= storage
            return True
        return False
    
    def release(self, cpu: int, ram: int, storage: int) -> None:
        """Releases compute resources"""
        self.available_cpu += cpu
        self.available_ram += ram
        self.available_storage += storage
    
    def usage_percentage(self) -> Tuple[float, float, float]:
        """Returns the usage percentage of resources (CPU, RAM, Storage)"""
        cpu_usage = ((self.cpu_cores - self.available_cpu) / self.cpu_cores) * 100
        ram_usage = ((self.ram - self.available_ram) / self.ram) * 100
        storage_usage = ((self.storage - self.available_storage) / self.storage) * 100
        return (cpu_usage, ram_usage, storage_usage)

class Node:
    def __init__(self, 
                 name: str, 
                 node_type: NodeType, 
                 resources: ComputeResources,
                 location: Tuple[float, float],  # (latitude, longitude)
                 bandwidth: float,  # Mbps
                 parent_latency: float = 0.0,  # ms - latency with parent node
                 sibling_latency: Dict[str, float] = None  # ms - latency with sibling nodes
                ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.node_type = node_type
        self.resources = resources
        self.location = location
        self.bandwidth = bandwidth
        self.parent_latency = parent_latency
        self.sibling_latency = sibling_latency or {}
        self.state = NodeState.RUNNING
        self.parent = None
        self.children = []
        self.running_services = {}  # service_id -> service
        self.event_queue = []  # List of events to be processed by this node
        self.stored_data = {}  # data_id -> data
    
    def add_child(self, child_node: 'Node') -> None:
        """Adds a child node"""
        self.children.append(child_node)
        child_node.parent = self
    
    def add_sibling_latency(self, sibling_id: str, latency: float) -> None:
        """Adds latency to a sibling node"""
        self.sibling_latency[sibling_id] = latency
    
    def calculate_latency_to(self, target_node_id: str) -> float:
        """Calculate latency to another node"""
        # If it's a sibling node, we have direct latency
        if target_node_id in self.sibling_latency:
            return self.sibling_latency[target_node_id]
        
        # If it's the parent
        if self.parent and self.parent.id == target_node_id:
            return self.parent_latency
        
        # If it's a child
        for child in self.children:
            if child.id == target_node_id:
                return child.parent_latency  # Latency from child to us (parent)
        
        # Otherwise, we need to go through the parent (simple approximation)
        if self.parent:
            return self.parent_latency + self.parent.calculate_latency_to(target_node_id)
        
        # Node not found
        return float('inf')
    
    def can_host_service(self, service) -> bool:
        """Checks if the node can host a given service"""
        return self.resources.allocate(service.cpu_required, service.ram_required, service.storage_required)
    
    def deploy_service(self, service) -> bool:
        """Deploy a service on this node"""
        if service.id in self.running_services:
            return True  # Arready deployed
        
        if self.can_host_service(service):
            self.running_services[service.id] = service
            service.node_id = self.id
            return True
        return False
    
    def remove_service(self, service_id: str) -> bool:
        """Removes a service from this node"""
        if service_id in self.running_services:
            service = self.running_services[service_id]
            self.resources.release(service.cpu_required, service.ram_required, service.storage_required)
            del self.running_services[service_id]
            return True
        return False
    
    def process_event(self, event, current_time: float) -> List:
        """Process an event and potentially generate new events"""

        new_events = []  # Initialize the list of new events
        
        # Check if the node is in a running state
        if self.state != NodeState.RUNNING:
            # If the node is down, do not process the event
            return new_events
        
        # TODO: Implement event processing logic
        # Based on event type (data processing, storage, etc.)
        

        if event.type == 'data_processing':

            pass
        elif event.type == 'storage':

            pass
        
        return new_events
    
    def store_data(self, data) -> bool:
        """Stores data on this node"""
        if self.resources.available_storage >= data.storage_size:
            self.stored_data[data.id] = data
            self.resources.available_storage -= data.storage_size
            return True
        return False
    
    def remove_data(self, data_id: str) -> bool:
        """Removes data from this node"""
        if data_id in self.stored_data:
            data = self.stored_data[data_id]
            self.resources.available_storage += data.storage_size
            del self.stored_data[data_id]
            return True
        return False
    
    def set_state(self, new_state: NodeState) -> None:
        """Changes the state of the node"""
        old_state = self.state
        self.state = new_state
        
        # Actions to perform when changing state
        if new_state == NodeState.FAILED:
            # Handle node failure (propagate to services, etc.)
            pass
        elif old_state == NodeState.FAILED and new_state == NodeState.RUNNING:
            # Handle recovery from failure
            pass