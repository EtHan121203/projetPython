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
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

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
        self.priority = priority
        self.created_at = time.time()
        self.ttl = ttl
        self.encrypted = encrypted
        self.processed_by = []  # List of service IDs that processed this data
        self.current_location = source_id  # Node ID where data is currently stored
    
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
        """Update the current location of the data"""
        self.current_location = node_id
    
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
            "priority": self.priority.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "encrypted": self.encrypted,
            "processed_by": self.processed_by,
            "current_location": self.current_location
        }
    
    @classmethod
    def deserialize(cls, data_dict: Dict) -> 'Data':
        """Create a Data object from serialized data"""
        data = cls(
            data_type=DataType(data_dict["type"]),
            content=data_dict["content"],
            source_id=data_dict["source"],
            storage_size=data_dict["size"],
            priority=DataPriority(data_dict["priority"]),
            ttl=data_dict["ttl"],
            encrypted=data_dict["encrypted"]
        )
        data.id = data_dict["id"]
        data.created_at = data_dict["created_at"]
        data.processed_by = data_dict["processed_by"]
        data.current_location = data_dict["current_location"]
        return data
    
    def __str__(self) -> str:
        return f"Data({self.id}, {self.data_type.value}, size={self.storage_size} bytes, priority={self.priority.value})"