import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from models.node import Node, NodeType, ComputeResources
from models.sensor import Sensor, SensorType
from models.actuator import Actuator, ActuatorType
from models.service import Service, ServiceType
from models.data import DataType
from simulator.network import Network

class ConfigLoader:
    """
    Utility for loading simulator configurations from files.
    Handles loading node configurations, network topology, and simulation parameters.
    """
    def __init__(self):
        self.logger = logging.getLogger("ConfigLoader")
    
    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {file_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration from {file_path}: {e}")
            raise
    
    def create_network_from_config(self, config: Dict[str, Any]) -> Network:
        """
        Create a network topology from a configuration dictionary
        
        Args:
            config: Dictionary containing network configuration
            
        Returns:
            Configured Network object
        """
        network = Network()
        
        # Create nodes
        if "nodes" in config:
            for node_data in config["nodes"]:
                node = self._create_node_from_config(node_data)
                network.add_node(node)
        
        # Create connections
        if "connections" in config:
            for conn_data in config["connections"]:
                from_node = conn_data.get("from")
                to_node = conn_data.get("to")
                bandwidth = conn_data.get("bandwidth", 100.0)  # Default 100 Mbps
                latency = conn_data.get("latency", 10.0)  # Default 10 ms
                
                if from_node and to_node:
                    network.connect(from_node, to_node, bandwidth, latency)
        
        return network
    
    def _create_node_from_config(self, node_data: Dict[str, Any]) -> Node:
        """
        Create a node from configuration data
        
        Args:
            node_data: Dictionary containing node configuration
            
        Returns:
            Configured Node object
        """
        # Extract node properties
        name = node_data.get("name", "unnamed_node")
        node_type_str = node_data.get("type", "edge")
        
        # Parse node type
        try:
            node_type = NodeType[node_type_str.upper()]
        except KeyError:
            self.logger.warning(f"Unknown node type '{node_type_str}', defaulting to EDGE")
            node_type = NodeType.EDGE
        
        # Set default resource values based on node type
        default_resources = {
            NodeType.EDGE: {"cpu_cores": 2, "cpu_frequency": 1.5, "ram": 2048, "storage": 16000},
            NodeType.FOG: {"cpu_cores": 8, "cpu_frequency": 2.5, "ram": 16384, "storage": 32000},
            NodeType.CLOUD: {"cpu_cores": 32, "cpu_frequency": 3.0, "ram": 65536, "storage": 1000000}
        }
        
        # Extract node resources
        resources_data = node_data.get("resources", {})
        cpu_cores = resources_data.get("cpu_cores", default_resources[node_type]["cpu_cores"])
        cpu_frequency = resources_data.get("cpu_frequency", default_resources[node_type]["cpu_frequency"])
        ram = resources_data.get("ram", default_resources[node_type]["ram"])
        storage = resources_data.get("storage", default_resources[node_type]["storage"])
        
        # Create resources object
        resources = ComputeResources(
            cpu_cores=cpu_cores,
            cpu_frequency=cpu_frequency,
            ram=ram,
            storage=storage
        )
        
        # Extract location
        location_data = node_data.get("location", [0.0, 0.0])
        location = (location_data[0], location_data[1])
        
        # Extract network parameters
        bandwidth = node_data.get("bandwidth", 100.0)  # Default 100 Mbps
        parent_latency = node_data.get("parent_latency", 5.0)  # Default 5 ms
        
        # Create the node
        node = Node(
            name=name,
            node_type=node_type,
            resources=resources,
            location=location,
            bandwidth=bandwidth,
            parent_latency=parent_latency
        )
        
        return node
    
    def create_sensors_from_config(self, config: Dict[str, Any], network: Network) -> List[Sensor]:
        """
        Create sensors from a configuration dictionary
        
        Args:
            config: Dictionary containing sensor configurations
            network: Network instance to add sensors to
            
        Returns:
            List of created Sensor objects
        """
        sensors = []
        sensor_configs = config.get("sensors", [])
        
        # Create sensors
        for sensor_data in sensor_configs:
            # Extract sensor properties
            name = sensor_data.get("name", f"sensor_{len(sensors)}")
            sensor_type_str = sensor_data.get("type", "generic")
            
            # Parse sensor type
            try:
                sensor_type = SensorType[sensor_type_str.upper()]
            except KeyError:
                self.logger.warning(f"Unknown sensor type '{sensor_type_str}', defaulting to GENERIC")
                sensor_type = SensorType.GENERIC
            
            # Extract location if available
            location = None
            if "location" in sensor_data:
                location = (sensor_data["location"][0], sensor_data["location"][1])
            
            # Create the sensor
            sensor = Sensor(
                name=name,
                sensor_type=sensor_type,
                data_type=DataType[sensor_data.get("data_type", "TEMPERATURE").upper()],
                generation_rate=sensor_data.get("generation_rate", 1.0),  # 1 data point per second by default
                location=location
            )
            
            # Attach to node if specified
            if "node_id" in sensor_data:
                node_id = sensor_data["node_id"]
                if node_id in network.nodes:
                    try:
                        sensor.attach_to_node(node_id)
                        self.logger.info(f"Sensor {name} attached to node {node_id}")
                        
                        # Also set sensor location to match node if sensor doesn't have one
                        if not location and hasattr(network.nodes[node_id], 'location'):
                            sensor.location = network.nodes[node_id].location
                            self.logger.debug(f"Sensor {name} location set to match node {node_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to attach sensor {name} to node {node_id}: {str(e)}")
                else:
                    self.logger.warning(f"Sensor {name} references non-existent node {node_id}")
                    
                    # Try to find a suitable node based on location if provided
                    if location:
                        closest_node = None
                        min_distance = float('inf')
                        
                        for node_id, node in network.nodes.items():
                            if not hasattr(node, 'location'):
                                continue
                            
                            distance = ((location[0] - node.location[0]) ** 2 + 
                                       (location[1] - node.location[1]) ** 2) ** 0.5
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_node = node_id
                        
                        if closest_node:
                            sensor.attach_to_node(closest_node)
                            self.logger.info(f"Sensor {name} auto-attached to closest node {closest_node}")
            
            sensors.append(sensor)
        
        return sensors
    
    def create_actuators_from_config(self, config: Dict[str, Any], network: Network) -> List[Actuator]:
        """
        Create actuators from a configuration dictionary
        
        Args:
            config: Dictionary containing actuator configurations
            network: Network instance to add actuators to
            
        Returns:
            List of created Actuator objects
        """
        actuators = []
        
        if "actuators" in config:
            for actuator_data in config["actuators"]:
                name = actuator_data.get("name", "unnamed_actuator")
                
                # Parse actuator type
                type_str = actuator_data.get("type", "traffic_light")
                try:
                    actuator_type = ActuatorType[type_str.upper()]
                except KeyError:
                    self.logger.warning(f"Unknown actuator type '{type_str}', defaulting to TRAFFIC_LIGHT")
                    actuator_type = ActuatorType.TRAFFIC_LIGHT
                
                # Extract actuator properties
                location_data = actuator_data.get("location", [0.0, 0.0])
                location = (location_data[0], location_data[1])
                power_consumption = actuator_data.get("power_consumption", 1.0)  # Default 1 W
                response_time = actuator_data.get("response_time", 0.1)  # Default 0.1s
                reliability = actuator_data.get("reliability", 0.99)  # Default 99% reliable
                
                # Create the actuator
                actuator = Actuator(
                    name=name,
                    actuator_type=actuator_type,
                    location=location,
                    power_consumption=power_consumption,
                    response_time=response_time,
                    reliability=reliability
                )
                
                # Attach to node if specified
                if "node_id" in actuator_data:
                    node_id = actuator_data["node_id"]
                    if node_id in network.nodes:
                        actuator.attach_to_node(node_id)
                    else:
                        self.logger.warning(f"Actuator {name} references non-existent node {node_id}")
                
                actuators.append(actuator)
        
        return actuators
    
    def create_services_from_config(self, config: Dict[str, Any], network: Network) -> List[Service]:
        """
        Create services from a configuration dictionary
        
        Args:
            config: Dictionary containing service configurations
            network: Network instance to deploy services on
            
        Returns:
            List of created Service objects
        """
        services = []
        
        if "services" in config:
            for service_data in config["services"]:
                name = service_data.get("name", "unnamed_service")
                
                # Parse service type
                type_str = service_data.get("type", "data_analytics")
                try:
                    service_type = ServiceType[type_str.upper()]
                except KeyError:
                    self.logger.warning(f"Unknown service type '{type_str}', defaulting to DATA_ANALYTICS")
                    service_type = ServiceType.DATA_ANALYTICS
                
                # Extract service resource requirements
                cpu_required = service_data.get("cpu_required", 1)
                ram_required = service_data.get("ram_required", 512)
                storage_required = service_data.get("storage_required", 1000)
                
                # Extract input and output data types
                input_types_str = service_data.get("input_types", ["SENSOR_DATA"])
                output_types_str = service_data.get("output_types", [])
                
                input_types = set()
                for type_str in input_types_str:
                    try:
                        input_types.add(DataType[type_str.upper()])
                    except KeyError:
                        self.logger.warning(f"Unknown data type '{type_str}', skipping")
                
                output_types = set()
                for type_str in output_types_str:
                    try:
                        output_types.add(DataType[type_str.upper()])
                    except KeyError:
                        self.logger.warning(f"Unknown data type '{type_str}', skipping")
                
                # Extract processing time parameters
                min_processing_time = service_data.get("min_processing_time", 0.1)
                max_processing_time = service_data.get("max_processing_time", 1.0)
                priority = service_data.get("priority", 0)
                
                # Create the service
                service = Service(
                    name=name,
                    service_type=service_type,
                    cpu_required=cpu_required,
                    ram_required=ram_required,
                    storage_required=storage_required,
                    input_types=input_types,
                    output_types=output_types,
                    min_processing_time=min_processing_time,
                    max_processing_time=max_processing_time,
                    priority=priority
                )
                
                # Deploy to node if specified
                if "node_id" in service_data:
                    node_id = service_data["node_id"]
                    if node_id in network.nodes:
                        service.deploy(node_id)
                        # Add service to node
                        network.nodes[node_id].deploy_service(service)
                    else:
                        self.logger.warning(f"Service {name} references non-existent node {node_id}")
                
                services.append(service)
        
        return services
    
    def save_config(self, config: Dict[str, Any], file_path: str) -> bool:
        """
        Save configuration to a JSON file
        
        Args:
            config: Dictionary containing configuration settings
            file_path: Path to save the configuration file
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"Saved configuration to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration to {file_path}: {e}")
            return False
    
    def generate_sample_config(self, file_path: str) -> bool:
        """
        Generate a sample configuration file
        
        Args:
            file_path: Path to save the sample configuration file
            
        Returns:
            True if generation was successful, False otherwise
        """
        sample_config = {
            "simulation": {
                "max_time": 3600,
                "log_level": "INFO"
            },
            "nodes": [
                {
                    "name": "cloud_server",
                    "type": "cloud",
                    "location": [0.0, 0.0],
                    "resources": {
                        "cpu_cores": 32,
                        "cpu_frequency": 3.0,
                        "ram": 65536,
                        "storage": 1000000
                    },
                    "bandwidth": 1000.0
                },
                {
                    "name": "fog_node_1",
                    "type": "fog",
                    "location": [1.0, 1.0],
                    "resources": {
                        "cpu_cores": 8,
                        "cpu_frequency": 2.5,
                        "ram": 16384,
                        "storage": 32000
                    },
                    "bandwidth": 500.0,
                    "parent_latency": 20.0
                },
                {
                    "name": "edge_node_1",
                    "type": "edge",
                    "location": [1.5, 1.5],
                    "resources": {
                        "cpu_cores": 2,
                        "cpu_frequency": 1.5,
                        "ram": 2048,
                        "storage": 16000
                    },
                    "bandwidth": 100.0,
                    "parent_latency": 5.0
                }
            ],
            "connections": [
                {
                    "from": "cloud_server",
                    "to": "fog_node_1",
                    "bandwidth": 1000.0,
                    "latency": 20.0
                },
                {
                    "from": "fog_node_1",
                    "to": "edge_node_1",
                    "bandwidth": 100.0,
                    "latency": 5.0
                }
            ],
            "sensors": [
                {
                    "name": "temp_sensor_1",
                    "type": "temperature",
                    "location": [1.5, 1.6],
                    "node_id": "edge_node_1",
                    "sampling_rate": 0.1,
                    "data_size": 20
                },
                {
                    "name": "traffic_sensor_1",
                    "type": "traffic",
                    "location": [1.6, 1.5],
                    "node_id": "edge_node_1",
                    "sampling_rate": 1.0,
                    "data_size": 500
                }
            ],
            "actuators": [
                {
                    "name": "traffic_light_1",
                    "type": "traffic_light",
                    "location": [1.55, 1.55],
                    "node_id": "edge_node_1",
                    "response_time": 0.2
                }
            ],
            "services": [
                {
                    "name": "traffic_monitoring",
                    "type": "traffic_management",
                    "node_id": "fog_node_1",
                    "cpu_required": 2,
                    "ram_required": 4096,
                    "storage_required": 8000,
                    "input_types": ["SENSOR_DATA"],
                    "output_types": ["COMMAND"]
                },
                {
                    "name": "data_analytics",
                    "type": "data_analytics",
                    "node_id": "cloud_server",
                    "cpu_required": 8,
                    "ram_required": 16384,
                    "storage_required": 32000,
                    "input_types": ["SENSOR_DATA", "SYSTEM_DATA"],
                    "output_types": ["ANALYTICS"]
                }
            ]
        }
        
        return self.save_config(sample_config, file_path)

# Helper function to create a ConfigLoader
def create_config_loader():
    return ConfigLoader()