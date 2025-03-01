# simulator/discrete_event_simulator.py
import heapq
import logging
from typing import Dict, List, Any, Callable, Optional
import time as system_time

from models.event import Event, EventType
from models.node import Node, NodeState
from models.service import ServiceState  # <-- Import ServiceState here, not from node.
from models.sensor import Sensor
from models.actuator import Actuator
from models.service import Service
from models.data import Data, DataStatus
from simulator.scheduler import Scheduler

class DiscreteEventSimulator:
    def __init__(self, max_simulation_time: float = float('inf')):
        # Simulation time
        self.current_time = 0.0
        self.max_simulation_time = max_simulation_time
        self.is_running = False
        self.is_paused = False
        
        # Event queue
        self.event_queue = []
        
        # Simulation entities
        self.nodes: Dict[str, Node] = {}
        self.sensors: Dict[str, Sensor] = {}
        self.actuators: Dict[str, Actuator] = {}
        self.services: Dict[str, Service] = {}
        self.data: Dict[str, Data] = {}
        
        # Event handlers by type
        self.event_handlers = {
            EventType.DATA_GENERATION: self._handle_data_generation,
            EventType.DATA_TRANSMISSION: self._handle_data_transmission,
            EventType.DATA_PROCESSING: self._handle_data_processing,
            EventType.DATA_STORAGE: self._handle_data_storage,
            EventType.DATA_DELETION: self._handle_data_deletion,
            EventType.SERVICE_START: self._handle_service_start,
            EventType.SERVICE_STOP: self._handle_service_stop,
            EventType.SERVICE_MIGRATION: self._handle_service_migration,
            EventType.SERVICE_FAILURE: self._handle_service_failure,
            EventType.NODE_FAILURE: self._handle_node_failure,
            EventType.NODE_RECOVERY: self._handle_node_recovery,
            EventType.NODE_OVERLOAD: self._handle_node_overload,
            EventType.ACTUATOR_ACTION: self._handle_actuator_action,
            EventType.SIMULATION_START: self._handle_simulation_start,
            EventType.SIMULATION_STOP: self._handle_simulation_stop,
            EventType.SIMULATION_PAUSE: self._handle_simulation_pause,
            EventType.SIMULATION_RESUME: self._handle_simulation_resume,
            EventType.SIMULATION_STEP: self._handle_simulation_step,
            EventType.PERIODIC_CHECK: self._handle_periodic_check,
            EventType.CUSTOM: self._handle_custom_event
        }
        
        # Simulation metrics
        self.metrics = {
            "processed_events": 0,
            "total_data_generated": 0,
            "total_data_transmitted": 0,
            "total_data_processed": 0,
            "total_service_starts": 0,
            "total_service_stops": 0,
            "total_node_failures": 0,
            "start_time": 0.0,
            "end_time": 0.0,
            "real_time_elapsed": 0.0,
            "events_by_type": {},
            "node_resource_usage": {},
            "service_processing_times": {},
            "network_latencies": [],
            "response_times": []
        }
        
        # Configuration and observers
        self.config = {}
        self.observers = []
        
        # Logging
        self.logger = logging.getLogger("SimulatorLogger")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.scheduler = Scheduler()
    
    # ==== Simulation management methods ====
    
    def schedule_event(self, event: Event) -> None:
        """Schedules an event in the event queue"""
        heapq.heappush(self.event_queue, event)
        # Update metrics
        event_type = event.event_type.value
        if event_type not in self.metrics["events_by_type"]:
            self.metrics["events_by_type"][event_type] = 0
        self.metrics["events_by_type"][event_type] += 1
    
    def run(self, max_steps: int = None) -> None:
        """Runs the simulation until complete or until max_steps"""
        if not self.is_running:
            self._start_simulation()
        
        step_count = 0
        self.is_paused = False
        
        start_real_time = system_time.time()
        
        try:
            while (self.event_queue and 
                   self.current_time <= self.max_simulation_time and 
                   (max_steps is None or step_count < max_steps) and
                   not self.is_paused):
                
                # Process the next event
                next_event = heapq.heappop(self.event_queue)
                
                # Update the current time
                self.current_time = next_event.scheduled_time
                
                # Process the event
                self._process_event(next_event)
                
                # Increment the step counter
                step_count += 1
                
                # Notify observers
                self._notify_observers(next_event)
            
            # End the simulation if the event queue is empty or the maximum simulation time is reached
            if not self.event_queue or self.current_time > self.max_simulation_time:
                self._stop_simulation()
        
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise
        finally:
            end_real_time = system_time.time()
            self.metrics["real_time_elapsed"] += (end_real_time - start_real_time)
    
    def step(self) -> Optional[Event]:
        """Executes a simulation step and returns the processed event"""
        if not self.is_running:
            self._start_simulation()
        
        if not self.event_queue or self.current_time > self.max_simulation_time:
            self._stop_simulation()
            return None
        
        # Process the next event
        next_event = heapq.heappop(self.event_queue)
        
        # Update the current time
        self.current_time = next_event.scheduled_time
        
        # Process the event
        self._process_event(next_event)
        
        # Notify observers
        self._notify_observers(next_event)
        
        return next_event
    
    def pause(self) -> None:
        """Pauses the simulation"""
        self.is_paused = True
        self.logger.info(f"Simulation paused at t={self.current_time}")
    
    def resume(self) -> None:
        """Resumes the simulation after a pause"""
        self.is_paused = False
        self.logger.info(f"Simulation resumed at t={self.current_time}")
        self.run()
    
    def reset(self) -> None:
        """Resets the simulation"""
        self.current_time = 0.0
        self.is_running = False
        self.is_paused = False
        self.event_queue = []
        
        # Reset metrics
        self.metrics = {
            "processed_events": 0,
            "total_data_generated": 0,
            "total_data_transmitted": 0,
            "total_data_processed": 0,
            "total_service_starts": 0,
            "total_service_stops": 0,
            "total_node_failures": 0,
            "start_time": 0.0,
            "end_time": 0.0,
            "real_time_elapsed": 0.0,
            "events_by_type": {},
            "node_resource_usage": {},
            "service_processing_times": {},
            "network_latencies": [],
            "response_times": []
        }
        
        # Reset the state of entities
        for node in self.nodes.values():
            node.state = NodeState.RUNNING
            node.resources.available_cpu = node.resources.cpu_cores
            node.resources.available_ram = node.resources.ram
            node.resources.available_storage = node.resources.storage
            node.running_services = {}
            node.stored_data = {}
        
        for service in self.services.values():
            service.state = NodeState.STOPPED
            service.node_id = None
            service.processed_data_count = 0
            service.processing_queue = []
            service.current_processing = None
        
        self.logger.info("Simulation reset")
    
    def add_node(self, node: Node) -> None:
        """Adds a node to the simulation"""
        self.nodes[node.id] = node
        self.metrics["node_resource_usage"][node.id] = []
    
    def add_sensor(self, sensor: Sensor) -> None:
        """Adds a sensor to the simulation"""
        self.sensors[sensor.id] = sensor
        
        # Schedule the first data generation event
        if sensor.generation_frequency > 0:
            first_gen_time = self.current_time + (1.0 / sensor.generation_frequency)
            event = Event.create(
                event_type=EventType.DATA_GENERATION,
                creation_time=self.current_time,
                scheduled_time=first_gen_time,
                source_id=sensor.id,
                target_ids=[sensor.gateway_id]
            )
            self.schedule_event(event)
    
    def add_actuator(self, actuator: Actuator) -> None:
        """Adds an actuator to the simulation"""
        self.actuators[actuator.id] = actuator
    
    def add_service(self, service: Service) -> None:
        """Adds a service to the simulation"""
        self.services[service.id] = service
        self.metrics["service_processing_times"][service.id] = []
    
    def add_observer(self, observer: Callable) -> None:
        """Adds an observer that will be notified after each event"""
        self.observers.append(observer)
    
    def set_logging_level(self, level: int) -> None:
        """Sets the logging level"""
        self.logger.setLevel(level)
    
    def get_metrics(self) -> Dict:
        """Returns the current simulation metrics"""
        # Update metrics with current values
        for node_id, node in self.nodes.items():
            cpu_usage, ram_usage, storage_usage = node.resources.usage_percentage()
            self.metrics["node_resource_usage"][node_id].append({
                "time": self.current_time,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "storage_usage": storage_usage
            })
        
        return self.metrics
    
    def get_event_count(self) -> int:
        """Returns the number of events in the event queue"""
        return len(self.event_queue)
    
    def set_speed_factor(self, speed_factor: float) -> None:
        """Sets the simulation speed multiplier"""
        self.scheduler.set_speed_factor(speed_factor)
    
    # ==== Private methods ====
    
    def _start_simulation(self) -> None:
        """Starts the simulation"""
        self.is_running = True
        self.metrics["start_time"] = self.current_time
        
        # Schedule a simulation start event
        start_event = Event.create(
            event_type=EventType.SIMULATION_START,
            creation_time=self.current_time,
            scheduled_time=self.current_time,
            source_id="simulator"
        )
        self.schedule_event(start_event)
        
        # Schedule periodic checks
        check_event = Event.create(
            event_type=EventType.PERIODIC_CHECK,
            creation_time=self.current_time,
            scheduled_time=self.current_time + 1.0,  # First check after 1 second
            source_id="simulator"
        )
        self.schedule_event(check_event)
        
        self.logger.info(f"Simulation started at t={self.current_time}")
    
    def _stop_simulation(self) -> None:
        """Stops the simulation"""
        self.is_running = False
        self.metrics["end_time"] = self.current_time
        
        # Schedule a simulation stop event
        stop_event = Event.create(
            event_type=EventType.SIMULATION_STOP,
            creation_time=self.current_time,
            scheduled_time=self.current_time,
            source_id="simulator"
        )
        self.schedule_event(stop_event)
        
        self.logger.info(f"Simulation stopped at t={self.current_time}")
    
    def _process_event(self, event: Event) -> None:
        """Processes an event"""
        start_time = system_time.time()
        
        # Retrieve the appropriate event handler
        handler = event.handler or self.event_handlers.get(event.event_type)
        
        if handler:
            # Call the event handler
            try:
                new_events = handler(event)
                if new_events:
                    for new_event in new_events:
                        self.schedule_event(new_event)
            except Exception as e:
                self.logger.error(f"Error processing event {event.id}: {str(e)}")
                # Continue the simulation despite the error
        else:
            self.logger.warning(f"No handler for event type {event.event_type}")
        
        # Calculate the actual processing time
        end_time = system_time.time()
        processing_time = end_time - start_time
        
        # Mark the event as processed
        event.mark_processed(processing_time)
        
        # Update metrics
        self.metrics["processed_events"] += 1
    
    def _notify_observers(self, event: Event) -> None:
        """Notifies all observers after processing an event"""
        for observer in self.observers:
            try:
                observer(event, self.current_time, self)
            except Exception as e:
                self.logger.error(f"Error in observer: {str(e)}")
    
    # ==== Event Handlers ====
    
    def _handle_data_generation(self, event: Event) -> List[Event]:
        """Handles a data generation event"""
        new_events = []
        
        sensor_id = event.source_id
        if sensor_id not in self.sensors:
            self.logger.warning(f"Sensor {sensor_id} not found")
            return new_events
        
        sensor = self.sensors[sensor_id]
        
        # Generate data if the sensor is active
        if sensor.is_active:
            data_info = sensor.generate_data(self.current_time)
            
            if data_info:
                # Create the Data object
                data = Data(
                    data_type=data_info["data_type"],
                    value=data_info["value"],
                    storage_size=data_info["size"],
                    transmission_size=data_info["size"],
                    processing_load=1.0,
                    producer_id=sensor_id
                )
                data.set_creation_time(self.current_time)
                data.current_node_id = sensor.gateway_id
                
                # Add the data to the simulation
                self.data[data.id] = data
                
                # Update metrics
                self.metrics["total_data_generated"] += 1
                
                # Schedule a transmission event
                transmission_time = self.current_time + (sensor.network_latency / 1000.0)  # Convert ms to seconds
                transmission_event = Event.create(
                    event_type=EventType.DATA_TRANSMISSION,
                    creation_time=self.current_time,
                    scheduled_time=transmission_time,
                    source_id=sensor_id,
                    target_ids=[sensor.gateway_id],
                    data={"data_id": data.id}
                )
                new_events.append(transmission_event)
            
            # Schedule the next data generation
            next_gen_time = sensor.get_next_generation_time()
            next_gen_event = Event.create(
                event_type=EventType.DATA_GENERATION,
                creation_time=self.current_time,
                scheduled_time=next_gen_time,
                source_id=sensor_id,
                target_ids=[sensor.gateway_id]
            )
            new_events.append(next_gen_event)
        
        return new_events
    
    def _handle_data_transmission(self, event: Event) -> List[Event]:
        """Handles a data transmission event"""
        new_events = []
        
        data_id = event.data.get("data_id")
        if data_id not in self.data:
            self.logger.warning(f"Data {data_id} not found")
            return new_events
        
        data = self.data[data_id]
        source_id = event.source_id
        target_ids = event.target_ids
        
        # Update the status of the data
        data.update_status(DataStatus.TRANSMITTED, self.current_time)
        
        # Update metrics
        self.metrics["total_data_transmitted"] += 1
        
        # Schedule a processing event for each consuming service
        for target_id in target_ids:
            if target_id in self.services:
                service = self.services[target_id]
                processing_time = service.process_data(data)
                
                if processing_time > 0:
                    # Schedule a storage event after processing
                    processing_event = Event.create(
                        event_type=EventType.DATA_PROCESSING,
                        creation_time=self.current_time,
                        scheduled_time=self.current_time + processing_time,
                        source_id=service.id,
                        target_ids=[service.node_id],
                        data={"data_id": data.id}
                    )
                    new_events.append(processing_event)
        
        return new_events
    
    def _handle_data_processing(self, event: Event) -> List[Event]:
        """Handles a data processing event"""
        new_events = []
        
        data_id = event.data.get("data_id")
        if data_id not in self.data:
            self.logger.warning(f"Data {data_id} not found")
            return new_events
        
        data = self.data[data_id]
        service_id = event.source_id
        node_id = event.target_ids[0] if event.target_ids else None
        
        # Check if service exists
        if service_id not in self.services:
            self.logger.warning(f"Service {service_id} non trouvé")
            return new_events
        
        service = self.services[service_id]
        
        # Update data status
        data.update_status("processed", self.current_time)
        
        # Update metrics
        self.metrics["total_data_processed"] += 1
        
        # Schedule data storage event
        storage_time = self.current_time + 0.01  # Small delay for storage
        storage_event = Event.create(
            event_type=EventType.DATA_STORAGE,
            creation_time=self.current_time,
            scheduled_time=storage_time,
            source_id=service_id,
            target_ids=[node_id] if node_id else [],
            data={"data_id": data_id}
        )
        new_events.append(storage_event)
        
        return new_events

    def _handle_data_storage(self, event: Event) -> List[Event]:
        """Handles a data storage event"""
        new_events = []
        
        data_id = event.data.get("data_id")
        if data_id not in self.data:
            self.logger.warning(f"Data {data_id} not found")
            return new_events
        
        data = self.data[data_id]
        node_id = event.target_ids[0] if event.target_ids else None
        
        # Check if node exists
        if node_id not in self.network.nodes:
            self.logger.warning(f"Nœud {node_id} non trouvé")
            return new_events
        
        node = self.network.nodes[node_id]
        
        # Store data in node
        if node.store_data(data):
            # Update data status
            data.update_status("stored", self.current_time)
            
            # Update metrics
            self.metrics["total_data_stored"] += 1
        else:
            self.logger.warning(f"Échec du stockage de la donnée {data_id} sur le nœud {node_id}")
            
            # Schedule data deletion event after some time
            deletion_time = self.current_time + 1.0  # 1 second delay
            deletion_event = Event.create(
                event_type=EventType.DATA_DELETION,
                creation_time=self.current_time,
                scheduled_time=deletion_time,
                source_id=node_id,
                data={"data_id": data_id}
            )
            new_events.append(deletion_event)
        
        return new_events

    def _handle_data_deletion(self, event: Event) -> List[Event]:
        """Handles a data deletion event"""
        new_events = []
        
        data_id = event.data.get("data_id")
        if data_id in self.data:
            del self.data[data_id]
            self.logger.info(f"Data {data_id} has been deleted from simulator.")
            self.metrics["total_data_deleted"] += 1
        else:
            self.logger.warning(f"Data {data_id} not found for deletion.")
        
        return new_events

    def _handle_service_start(self, event: Event) -> List[Event]:
        """Handles a service start event"""
        new_events = []
        
        service_id = event.source_id
        node_id = event.target_ids[0] if event.target_ids else None
        
        if service_id not in self.services:
            self.logger.warning(f"Service {service_id} introuvable.")
            return new_events
        
        service = self.services[service_id]
        service.state = ServiceState.RUNNING
        self.logger.info(f"Service {service_id} démarré sur le nœud {node_id}.")
        
        return new_events

    def _handle_service_stop(self, event: Event) -> List[Event]:
        """Handles a service stop event"""
        new_events = []
        
        service_id = event.source_id
        if service_id not in self.services:
            self.logger.warning(f"Service {service_id} not found.")
            return new_events
        
        service = self.services[service_id]
        service.state = ServiceState.STOPPED
        self.logger.info(f"Service {service_id} stopped.")
        
        return new_events

    def _handle_service_migration(self, event: Event) -> List[Event]:
        """Handles a service migration event"""
        new_events = []
        
        service_id = event.source_id
        target_node_id = event.target_ids[0] if event.target_ids else None
        
        if service_id not in self.services:
            self.logger.warning(f"Service {service_id} not found for migration.")
            return new_events
        
        # Stop the service at the current location
        service = self.services[service_id]
        service.state = ServiceState.STOPPED
        self.logger.info(f"Service {service_id} stopped for migration.")
        
        # Migrate to the new node
        self.logger.info(f"Migrating service {service_id} to node {target_node_id}.")
        
        # Start the service on the new node
        service.state = ServiceState.RUNNING
        self.logger.info(f"Service {service_id} started on node {target_node_id} after migration.")
        
        return new_events

    def _handle_service_failure(self, event: Event) -> List[Event]:
        """Handles a service failure event."""
        new_events = []
        
        service_id = event.source_id
        if service_id not in self.services:
            self.logger.warning(f"Service {service_id} not found.")
            return new_events
        
        service = self.services[service_id]
        service.state = ServiceState.FAILED
        self.logger.info(f"Service {service_id} has failed.")
        
        return new_events

    def _handle_node_failure(self, event: Event) -> List[Event]:
        """Handles a node failure event."""
        new_events = []
        
        node_id = event.source_id
        if node_id not in self.nodes:
            self.logger.warning(f"Node {node_id} not found.")
            return new_events
        
        node = self.nodes[node_id]
        node.state = NodeState.FAILED
        self.logger.info(f"Node {node_id} has failed.")
        
        # Handle services running on the node
        for service_id, service in node.running_services.items():
            service.state = ServiceState.FAILED
            self.logger.info(f"Service {service_id} has failed due to node failure.")
        
        return new_events

    def _handle_node_recovery(self, event: Event) -> List[Event]:
        """Handles a node recovery event."""
        new_events = []
        
        node_id = event.source_id
        if node_id not in self.nodes:
            self.logger.warning(f"Node {node_id} not found.")
            return new_events
        
        node = self.nodes[node_id]
        node.state = NodeState.RUNNING
        self.logger.info(f"Node {node_id} has recovered.")
        
        return new_events

    def _handle_node_overload(self, event: Event) -> List[Event]:
        """Handles a node overload event."""
        new_events = []
        
        node_id = event.source_id
        if node_id not in self.nodes:
            self.logger.warning(f"Node {node_id} not found.")
            return new_events
        
        node = self.nodes[node_id]
        node.state = NodeState.OVERLOADED
        self.logger.info(f"Node {node_id} is overloaded.")
        
        return new_events

    def _handle_actuator_action(self, event: Event) -> List[Event]:
        """Handles an actuator action event."""
        new_events = []
        
        actuator_id = event.source_id
        if actuator_id not in self.actuators:
            self.logger.warning(f"Actuator {actuator_id} not found.")
            return new_events
        
        actuator = self.actuators[actuator_id]
        self.logger.info(f"Actuator {actuator_id} has performed an action.")
        
        return new_events

    def _handle_simulation_start(self, event: Event) -> List[Event]:
        """Handles a simulation start event"""
        new_events = []
        self.logger.info("Simulation starting...")
        return new_events

    def _handle_simulation_stop(self, event: Event) -> List[Event]:
        """Handles a simulation stop event"""
        new_events = []
        self.logger.info("Simulation stopping...")
        return new_events

    def _handle_simulation_pause(self, event: Event) -> List[Event]:
        """Handles a simulation pause event"""
        new_events = []
        self.pause()
        return new_events

    def _handle_simulation_resume(self, event: Event) -> List[Event]:
        """Handles a simulation resume event"""
        new_events = []
        self.resume()
        return new_events

    def _handle_simulation_step(self, event: Event) -> List[Event]:
        """Handles a simulation step event"""
        new_events = []
        self.step()
        return new_events

    def _handle_periodic_check(self, event: Event) -> List[Event]:
        """Gère un événement de vérification périodique"""
        new_events = []
        
        # Here, you can add system state checks,
        # such as checking node loads, service availability,
        # or anomaly detection.
        
        self.logger.info("Performing periodic check...")
        
        # Schedule the next periodic check
        next_check_time = self.current_time + 1.0  # Next check after 1 second
        next_check_event = Event.create(
            event_type=EventType.PERIODIC_CHECK,
            creation_time=self.current_time,
            scheduled_time=next_check_time,
            source_id="simulator"
        )
        new_events.append(next_check_event)

        return new_events

    def _handle_custom_event(self, event: Event) -> List[Event]:
        """Handles a custom event"""
        new_events = []
        self.logger.info(f"Handling custom event: {event.event_type}")
        return new_events
