# models/node.py
import uuid
from enum import Enum
from typing import Dict, Set
from models.data import DataType, DataPriority

class ServiceType(Enum):
    TRAFFIC_MANAGEMENT = "traffic_management"
    AIR_QUALITY_MONITORING = "air_quality_monitoring" 
    ENERGY_MANAGEMENT = "energy_management"
    PUBLIC_SAFETY = "public_safety"
    WASTE_MANAGEMENT = "waste_management"
    PARKING_MANAGEMENT = "parking_management"
    PUBLIC_TRANSPORT = "public_transport"
    WATER_MANAGEMENT = "water_management"
    LIGHTING_CONTROL = "lighting_control"
    DATA_ANALYTICS = "data_analytics"

class ServiceState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    MIGRATING = "migrating"

class Service:
    def __init__(self,
                 name: str,
                 service_type: ServiceType,
                 cpu_required: int,  # Number of CPU cores
                 ram_required: int,  # MB
                 storage_required: int,  # MB
                 input_types: Set[DataType],
                 output_types: Set[DataType] = None,
                 min_processing_time: float = 0.1,  # seconds
                 max_processing_time: float = 1.0,  # seconds
                 priority: int = 0  # Service priority (higher values = higher priority)
                ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.service_type = service_type
        self.cpu_required = cpu_required
        self.ram_required = ram_required
        self.storage_required = storage_required
        self.input_types = input_types
        self.output_types = output_types or set()
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.priority = priority
        
        # Runtime attributes
        self.state = ServiceState.STOPPED
        self.node_id = None  # ID of the node where service is deployed
        self.processed_data_count = 0  # Counter for processed data items
        self.total_processing_time = 0.0  # Total time spent processing
        self.subscribers = []  # Services that subscribe to outputs from this service
        
    def deploy(self, node_id: str) -> bool:
        """Deploy service to a node"""
        if self.state == ServiceState.RUNNING:
            return False
        self.node_id = node_id
        self.state = ServiceState.RUNNING
        return True
    
    def undeploy(self) -> bool:
        """Undeploy service from its node"""
        if self.state == ServiceState.STOPPED:
            return False
        self.node_id = None
        self.state = ServiceState.STOPPED
        return True
    
    def pause(self) -> bool:
        """Pause service execution"""
        if self.state != ServiceState.RUNNING:
            return False
        self.state = ServiceState.PAUSED
        return True
    
    def resume(self) -> bool:
        """Resume service execution"""
        if self.state != ServiceState.PAUSED:
            return False
        self.state = ServiceState.RUNNING
        return True
    
    def can_process(self, data) -> bool:
        """Check if service can process the given data type"""
        return data.data_type in self.input_types and self.state == ServiceState.RUNNING
    
    def process_data(self, data) -> float:
        """
        Process incoming data
        
        Returns:
            float: Time taken to process data in seconds
        """
        if not self.can_process(data):
            return 0.0
        
        # Calculate processing time based on data size and priority
        # This is a simple model that could be made more sophisticated
        size_factor = min(1.0, data.storage_size / 1024.0)  # Normalize size influence
        
        # Priority factor: higher priority data gets processed faster
        priority_factor = {
            DataPriority.LOW: 1.2,
            DataPriority.MEDIUM: 1.0,
            DataPriority.HIGH: 0.8,
            DataPriority.CRITICAL: 0.6
        }.get(data.priority, 1.0)
        
        # Calculate processing time
        processing_time = (
            self.min_processing_time + 
            (self.max_processing_time - self.min_processing_time) * 
            size_factor * priority_factor
        )
        
        # Update service statistics
        self.processed_data_count += 1
        self.total_processing_time += processing_time
        
        # Mark data as processed by this service
        data.add_processing_record(self.id)
        
        return processing_time
    
    def add_subscriber(self, service_id: str) -> None:
        """Add a service that subscribes to this service's outputs"""
        if service_id not in self.subscribers:
            self.subscribers.append(service_id)
    
    def remove_subscriber(self, service_id: str) -> bool:
        """Remove a subscriber service"""
        if service_id in self.subscribers:
            self.subscribers.remove(service_id)
            return True
        return False
    
    def get_average_processing_time(self) -> float:
        """Get the average processing time per data item"""
        if self.processed_data_count == 0:
            return 0.0
        return self.total_processing_time / self.processed_data_count
    
    def get_status(self) -> Dict:
        """Get service status information"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.service_type.value,
            "state": self.state.value,
            "node_id": self.node_id,
            "processed_data_count": self.processed_data_count,
            "avg_processing_time": self.get_average_processing_time(),
            "cpu_required": self.cpu_required,
            "ram_required": self.ram_required,
            "storage_required": self.storage_required
        }
    
    def __str__(self) -> str:
        return f"Service({self.name}, {self.service_type.value}, {self.state.value})"