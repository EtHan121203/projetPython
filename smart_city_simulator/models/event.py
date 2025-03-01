import uuid
from enum import Enum
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

class EventType(Enum):
    DATA_GENERATION = "data_generation"
    DATA_TRANSMISSION = "data_transmission"
    DATA_PROCESSING = "data_processing"
    DATA_STORAGE = "data_storage"
    DATA_DELETION = "data_deletion"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    SERVICE_MIGRATION = "service_migration"
    SERVICE_FAILURE = "service_failure"
    NODE_FAILURE = "node_failure"
    NODE_RECOVERY = "node_recovery"
    NODE_OVERLOAD = "node_overload"
    ACTUATOR_ACTION = "actuator_action"
    SIMULATION_START = "simulation_start"
    SIMULATION_STOP = "simulation_stop"
    SIMULATION_PAUSE = "simulation_pause" 
    SIMULATION_RESUME = "simulation_resume"
    SIMULATION_STEP = "simulation_step"
    PERIODIC_CHECK = "periodic_check"
    CUSTOM = "custom"

@dataclass(order=True)
class Event:
    """
    Represents an event in the discrete event simulator.
    The class is decorated with order=True to allow sorting in the priority queue.
    The sort_index field is used to sort events by occurrence time.
    """
    sort_index: float  # Used to sort events (occurrence time)
    event_type: EventType = field(compare=False)  # Event type
    creation_time: float = field(compare=False)  # Event creation time 
    scheduled_time: float = field(compare=False)  # Scheduled occurrence time
    source_id: str = field(compare=False)  # Source ID of the event
    target_ids: List[str] = field(default_factory=list, compare=False)  # Target IDs
    data: Dict[str, Any] = field(default_factory=dict, compare=False)  # Associated data
    handler: Optional[Callable] = field(default=None, compare=False)  # Specific event handler
    id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)  # Unique ID
    priority: int = field(default=0, compare=False)  # Priority (used when times are equal)
    processed: bool = field(default=False, compare=False)  # Indicates if event was processed
    processing_time: float = field(default=0.0, compare=False)  # Actual processing time
    
    def __post_init__(self):
        """Initialize the sort_index field for sorting in the priority queue"""

        # Sort by occurrence time, then by priority (higher values first)
        self.sort_index = (self.scheduled_time, -self.priority)
    
    def mark_processed(self, processing_time: float) -> None:
        """Mark the event as processed"""

        self.processed = True
        self.processing_time = processing_time
    
    @classmethod
    def create(cls, 
               event_type: EventType,
               creation_time: float,
               scheduled_time: float,
               source_id: str,
               target_ids: List[str] = None,
               data: Dict[str, Any] = None,
               handler: Callable = None,
               priority: int = 0) -> 'Event':
        
        """Factory method to create an event"""
        return cls(
            sort_index=0.0,  # will be recalculated in __post_init__
            event_type=event_type,
            creation_time=creation_time,
            scheduled_time=scheduled_time,
            source_id=source_id,
            target_ids=target_ids or [],
            data=data or {},
            handler=handler,
            priority=priority
        )
    
    def get_transmission_delay(self) -> float:
        """Returns the delay between creation and scheduled occurrence"""

        return self.scheduled_time - self.creation_time
    
    def get_processing_delay(self) -> float:
        """Returns the processing delay (if processed)"""
        if self.processed:
            return self.processing_time
        return 0.0
    
    def clone_with_new_time(self, new_scheduled_time: float) -> 'Event':
        """Creates a clone of this event with a new scheduled time"""
        return Event.create(
            event_type=self.event_type,
            creation_time=self.creation_time,  # Garde le temps de création original
            scheduled_time=new_scheduled_time,
            source_id=self.source_id,
            target_ids=self.target_ids.copy(),
            data=self.data.copy(),
            handler=self.handler,
            priority=self.priority
        )
    
    def to_dict(self) -> Dict:
        """Converts the event to a dictionary for serialization"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "creation_time": self.creation_time,
            "scheduled_time": self.scheduled_time,
            "source_id": self.source_id,
            "target_ids": self.target_ids,
            "data": self.data,
            "priority": self.priority,
            "processed": self.processed,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, event_dict: Dict) -> 'Event':
        """Creates an event from a dictionary"""
        event = cls(
            sort_index=0.0,  # Sera recalculé dans __post_init__
            event_type=EventType(event_dict["event_type"]),
            creation_time=event_dict["creation_time"],
            scheduled_time=event_dict["scheduled_time"],
            source_id=event_dict["source_id"],
            target_ids=event_dict["target_ids"],
            data=event_dict["data"],
            priority=event_dict["priority"]
        )
        
        event.id = event_dict["id"]
        event.processed = event_dict["processed"]
        event.processing_time = event_dict["processing_time"]
        
        return event