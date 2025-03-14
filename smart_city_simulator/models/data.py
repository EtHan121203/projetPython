import uuid
from typing import Any, Dict, Optional, List
from enum import Enum
import json
import time

class DataStatus(Enum):
    GENERATED = "generated"
    TRANSMITTED = "transmitted"
    PROCESSING = "processing"
    PROCESSED = "processed"
    STORED = "stored"
    DELETED = "deleted"

class DataType(Enum):
    SENSOR_DATA = "sensor_data"
    USER_DATA = "user_data"
    SYSTEM_DATA = "system_data"
    ANALYTICS = "analytics"
    COMMAND = "command"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"

class DataPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class Data:
    """
    Represents data generated and processed in the smart city simulator
    """
    def __init__(self,
                 data_type: DataType,
                 content: Dict,
                 source_id: str,
                 storage_size: int,  # bytes
                 priority: DataPriority = DataPriority.MEDIUM,
                 ttl: Optional[float] = None,  # Time-to-live in seconds
                 encrypted: bool = False
                ):
        self.id = str(uuid.uuid4())
        self.data_type = data_type
        self.content = content
        self.source_id = source_id
        self.storage_size = storage_size
        # Add transmission_size attribute - by default same as storage_size
        self.transmission_size = storage_size  
        self.priority = priority
        self.created_at = time.time()
        self.ttl = ttl
        self.encrypted = encrypted
        self.processed_by = []  # List of service IDs that processed this data
        self.current_node_id = source_id  # Node ID where data is currently stored
        self.status = DataStatus.GENERATED  # Initial status is generated
        self.status_history = [{
            "status": DataStatus.GENERATED,
            "time": self.created_at
        }]
    
    def set_creation_time(self, creation_time: float) -> None:
        """Set the creation time of the data"""
        self.created_at = creation_time
    
    def set_transmission_size(self, size: int) -> None:
        """Set the transmission size of the data (might differ from storage size due to compression)"""
        self.transmission_size = size
    
    def update_status(self, status_str: str, timestamp: float = None) -> None:
        """Update the status of the data"""
        if timestamp is None:
            timestamp = time.time()
        
        try:
            new_status = DataStatus[status_str.upper()]
            self.status = new_status
        except KeyError:
            # Handle custom status strings that don't match enum values
            if status_str == "stored":
                self.status = DataStatus.STORED
            elif status_str == "failed_storage":
                self.status = DataStatus.FAILED
            elif status_str == "processed":
                self.status = DataStatus.PROCESSED
            else:
                self.status = DataStatus.UNKNOWN
        
        self.status_history.append({
            "status": self.status,
            "time": timestamp
        })

    def is_expired(self, current_time: float) -> bool:
        """Check if data has expired based on TTL"""
        if self.ttl is None:
            return False
        return (current_time - self.created_at) > self.ttl
    
    def add_processing_record(self, service_id: str) -> None:
        """Record that data was processed by a service"""
        if service_id not in self.processed_by:
            self.processed_by.append(service_id)
    
    def update_location(self, node_id: str) -> None:
        """Update the current node location of the data"""
        if not node_id:
            raise ValueError("Node ID cannot be None when updating data location")
        
        self.current_node_id = node_id
    
    def get_processing_history(self) -> List[str]:
        """Get the list of services that processed this data"""
        return self.processed_by
    
    def get_age(self, current_time: float) -> float:
        """Get the age of data in seconds"""
        return current_time - self.created_at
    
    def serialize(self) -> Dict:
        """Serialize data for transmission or storage"""
        return {
            "id": self.id,
            "type": self.data_type.value,
            "content": self.content,
            "source": self.source_id,
            "size": self.storage_size,
            "transmission_size": self.transmission_size,
            "priority": self.priority.value if hasattr(self.priority, 'value') else self.priority,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "encrypted": self.encrypted,
            "processed_by": self.processed_by,
            "current_node_id": self.current_node_id,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "status_history": [{"status": h["status"].value if hasattr(h["status"], 'value') else h["status"], 
                               "time": h["time"]} for h in self.status_history]
        }
    
    @classmethod
    def deserialize(cls, data_dict: Dict) -> 'Data':
        """Create a Data object from serialized data"""
        data = cls(
            data_type=DataType(data_dict["type"]),
            content=data_dict["content"],
            source_id=data_dict["source"],
            storage_size=data_dict["size"],
            priority=int(DataPriority(data_dict["priority"])),
            ttl=data_dict["ttl"],
            encrypted=data_dict["encrypted"]
        )
        data.id = data_dict["id"]
        data.created_at = data_dict["created_at"]
        
        # Set transmission_size if available, otherwise use storage_size
        if "transmission_size" in data_dict:
            data.transmission_size = data_dict["transmission_size"]
            
        data.processed_by = data_dict["processed_by"]
        data.current_node_id = data_dict.get("current_node_id", data_dict.get("current_location", ""))
        return data
    
    def __str__(self) -> str:
        return f"Data({self.id}, {self.data_type.value}, size={self.storage_size} bytes, priority={self.priority.value if hasattr(self.priority, 'value') else self.priority})"