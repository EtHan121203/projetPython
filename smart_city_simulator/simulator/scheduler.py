import heapq
import time
from typing import List, Dict, Any, Optional, Callable
import logging

from models.event import Event, EventType

class Scheduler:
    """
    Event scheduler for the smart city simulator.
    Manages event queue, timing, and periodic events.
    """
    def __init__(self):
        self.event_queue = []  # Priority queue of events
        self.periodic_events = {}  # id -> (event_template, interval, next_time)
        self.current_time = 0.0
        self.speed_factor = 1.0  # Simulation speed multiplier
        self.logger = logging.getLogger("Scheduler")
    
    def schedule_event(self, event: Event) -> None:
        """Schedule an event to be processed at its scheduled time"""
        heapq.heappush(self.event_queue, event)
    
    def schedule_events(self, events: List[Event]) -> None:
        """Schedule multiple events at once"""
        for event in events:
            heapq.heappush(self.event_queue, event)
    
    def get_next_event(self) -> Optional[Event]:
        """Get the next event from the queue without removing it"""
        if not self.event_queue:
            return None
        return self.event_queue[0]
    
    def pop_next_event(self) -> Optional[Event]:
        """Get and remove the next event from the queue"""
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)
    
    def schedule_periodic_event(self, 
                                event_type: EventType,
                                interval: float, 
                                source_id: str,
                                target_ids: List[str] = None,
                                data: Dict[str, Any] = None,
                                start_time: float = None,
                                end_time: float = None,
                                priority: int = 0) -> str:
        """
        Schedule an event to occur periodically at specified intervals
        
        Args:
            event_type: Type of event to schedule
            interval: Time between event occurrences (seconds)
            source_id: Source ID for the events
            target_ids: Target IDs for the events
            data: Data to include with the events
            start_time: When to start scheduling events (default: current time)
            end_time: When to stop scheduling events (default: never)
            priority: Priority of the events (higher values = higher priority)
            
        Returns:
            ID of the periodic event registration
        """
        pe_id = f"pe_{id(event_type)}_{source_id}_{time.time_ns()}"  # Unique ID

        if start_time is None:
            start_time = self.current_time

        event_template = {
            "event_type": event_type,
            "source_id": source_id,
            "target_ids": target_ids or [],
            "data": data or {},
            "priority": priority
        }

        self.periodic_events[pe_id] = {
            "template": event_template,
            "interval": interval,
            "next_time": start_time,
            "end_time": end_time
        }
        self._schedule_next_occurrence(pe_id)
        return pe_id
    
    def cancel_periodic_event(self, pe_id: str) -> bool:
        """Cancel a periodic event by its ID"""
        if pe_id in self.periodic_events:
            del self.periodic_events[pe_id]
            return True
        return False
    
    def update_time(self, new_time: float) -> None:
        """Update the current simulation time"""
        self.current_time = new_time
        
    def set_speed_factor(self, speed_factor: float) -> None:
        """Set the simulation speed multiplier"""
        if speed_factor <= 0:
            self.logger.error("Speed factor must be positive")
            return
            
        self.speed_factor = speed_factor
        self.logger.info(f"Simulation speed set to {speed_factor}x")
    
    def process_due_periodic_events(self) -> None:
        """Process all periodic events that are due for scheduling"""
        for pe_id, pe_info in list(self.periodic_events.items()):
            if pe_info["next_time"] <= self.current_time:
                # Time to schedule the next occurrence
                self._schedule_next_occurrence(pe_id)
    
    def _schedule_next_occurrence(self, pe_id: str) -> None:
        """Schedule the next occurrence of a periodic event"""
        if pe_id not in self.periodic_events:
            return
            
        pe_info = self.periodic_events[pe_id]
        template = pe_info["template"]
        next_time = pe_info["next_time"]
        
        # Check if we've reached the end time
        if pe_info["end_time"] is not None and next_time > pe_info["end_time"]:
            del self.periodic_events[pe_id]  # Remove expired periodic event
            return
            
        # Create event for this occurrence
        event = Event.create(
            event_type=template["event_type"],
            creation_time=self.current_time,
            scheduled_time=next_time,
            source_id=template["source_id"],
            target_ids=template["target_ids"],
            data=template["data"].copy(),
            priority=template["priority"]
        )
        
        # Schedule the event
        self.schedule_event(event)
        
        # Update next time
        pe_info["next_time"] = next_time + pe_info["interval"]
        
    def get_queue_length(self) -> int:
        """Get the number of events in the queue"""
        return len(self.event_queue)
    
    def get_event_time_range(self) -> Dict[str, float]:
        """Get the time range of scheduled events"""
        if not self.event_queue:
            return {"earliest": self.current_time, "latest": self.current_time}
            
        earliest = self.event_queue[0].scheduled_time
        latest = max(event.scheduled_time for event in self.event_queue)
        
        return {"earliest": earliest, "latest": latest}
    
    def get_events_by_type(self) -> Dict[str, int]:
        """Get count of scheduled events by type"""
        counts = {}
        for event in self.event_queue:
            event_type = event.event_type.value
            if event_type not in counts:
                counts[event_type] = 0
            counts[event_type] += 1
        return counts

    def schedule_periodic_check(self, interval=1.0):
        """Schedule periodic system checks"""
        return self.scheduler.schedule_periodic_event(
            event_type=EventType.PERIODIC_CHECK,
            interval=interval,  # Use a reasonable interval like 1 second
            source_id="scheduler",
            priority=0  # Explicitly set priority
        )