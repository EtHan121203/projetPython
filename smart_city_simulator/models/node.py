# models/node.py
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class NodeType(Enum):
    EDGE = "edge"
    FOG = "fog"
    CLOUD = "cloud"

class NodeState(Enum):
    RUNNING = "running"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

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
        """Initialize available resources to total resources if not provided"""
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
        self.available_cpu = min(self.cpu_cores, self.available_cpu + cpu)
        self.available_ram = min(self.ram, self.available_ram + ram)
        self.available_storage = min(self.storage, self.available_storage + storage)
    
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
                 location: Tuple[float, float],
                 bandwidth: float = 100.0,  # Mbps
                 parent_latency: float = 10.0  # ms
                ):
        self.id = f"{node_type.value}_{name}"
        self.name = name
        self.node_type = node_type
        self.resources = resources
        self.location = location
        self.bandwidth = bandwidth
        self.parent_latency = parent_latency  # Latency to parent node
        
        # Runtime attributes
        self.state = NodeState.RUNNING
        self.running_services = {}  # service_id -> Service
        self.stored_data = {}  # data_id -> Data
        self.neighbors = []  # List of neighbor node IDs
        self.parent_id = None  # Parent node ID
        self.children = []  # List of child node IDs
    
    def deploy_service(self, service) -> bool:
        """Deploy a service on this node"""
        if service.id in self.running_services:
            return True  # Already deployed
        
        # Check if we have resources
        if self.resources.allocate(
            service.cpu_required,
            service.ram_required,
            service.storage_required
        ):
            self.running_services[service.id] = service
            return True
        
        return False
    
    def undeploy_service(self, service_id: str) -> bool:
        """Remove a service from this node"""
        if service_id in self.running_services:
            service = self.running_services[service_id]
            self.resources.release(
                service.cpu_required,
                service.ram_required,
                service.storage_required
            )
            del self.running_services[service_id]
            return True
        return False
    
    def store_data(self, data) -> bool:
        """Store data on this node"""
        if data.id in self.stored_data:
            return True  # Already stored
        
        if self.resources.allocate(0, 0, data.storage_size):
            self.stored_data[data.id] = data
            data.update_location(self.id)
            return True
        
        return False
    
    def remove_data(self, data_id: str) -> bool:
        """Remove data from this node"""
        if data_id in self.stored_data:
            data = self.stored_data[data_id]
            self.resources.release(0, 0, data.storage_size)
            del self.stored_data[data_id]
            return True
        return False
    
    def is_overloaded(self) -> bool:
        """Check if the node is overloaded"""
        cpu_usage, ram_usage, storage_usage = self.resources.usage_percentage()
        return cpu_usage > 90 or ram_usage > 90 or storage_usage > 90
    
    def get_available_resources(self) -> Dict:
        """Get available resources"""
        return {
            "cpu": self.resources.available_cpu,
            "ram": self.resources.available_ram,
            "storage": self.resources.available_storage
        }
    
    def get_status(self) -> Dict:
        """Get node status information"""
        cpu_usage, ram_usage, storage_usage = self.resources.usage_percentage()
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "state": self.state.value,
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "storage_usage": storage_usage,
            "service_count": len(self.running_services),
            "stored_data_count": len(self.stored_data),
            "location": self.location
        }
    
    def add_child(self, node) -> None:
        """Add a child node to this node"""
        if node.id not in self.children:
            self.children.append(node.id)
            node.parent_id = self.id
            # Add to neighbors if not already there
            if node.id not in self.neighbors:
                self.neighbors.append(node.id)
            if self.id not in node.neighbors:
                node.neighbors.append(self.id)
    
    def add_neighbor(self, node) -> None:
        """Add a neighbor node to this node"""
        if node.id not in self.neighbors:
            self.neighbors.append(node.id)
        if self.id not in node.neighbors:
            node.neighbors.append(self.id)
    
    def __str__(self) -> str:
        return f"Node({self.name}, {self.node_type.value}, {self.state.value})"