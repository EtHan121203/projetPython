#!/usr/bin/env python3
import sys
import os
import logging
import argparse
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulator.log')
    ]
)
logger = logging.getLogger("Main")

# Import simulator components
from simulator.discrete_event_simulator import DiscreteEventSimulator
from simulator.network import Network
from simulator.scheduler import Scheduler
from models.node import Node, NodeType, ComputeResources
from models.sensor import Sensor, SensorType
from models.actuator import Actuator, ActuatorType
from models.service import Service, ServiceType
from models.data import DataType, DataPriority
from ui.gui import create_gui
from utils.config_loader import create_config_loader
from utils.statistics import create_statistics

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart City Simulator")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file")
    parser.add_argument("-s", "--sample-config", action="store_true", help="Generate sample configuration file")
    parser.add_argument("-g", "--gui", action="store_true", help="Launch GUI")
    parser.add_argument("-t", "--max-time", type=float, default=3600, help="Maximum simulation time (seconds)")
    parser.add_argument("-o", "--output", type=str, help="Output statistics file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def setup_simulator(args):
    """Set up the simulation environment based on arguments"""
    logger.info("Setting up simulator...")
    
    # Create simulator components
    simulator = DiscreteEventSimulator(max_simulation_time=args.max_time)
    config_loader = create_config_loader()
    statistics = create_statistics()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        simulator.set_logging_level(logging.DEBUG)
    
    # Generate sample config if requested
    if args.sample_config:
        sample_path = args.config or "sample_config.json"
        logger.info(f"Generating sample configuration to {sample_path}")
        if config_loader.generate_sample_config(sample_path):
            logger.info(f"Sample configuration generated at {sample_path}")
            if not args.config:  # Exit if only generating sample
                return None
    
    # Load configuration if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        try:
            config = config_loader.load_config(args.config)
            
            # Create network from config
            network = config_loader.create_network_from_config(config)
            simulator.network = network
            
            # Create sensors from config
            sensors = config_loader.create_sensors_from_config(config, network)
            for sensor in sensors:
                simulator.add_sensor(sensor)
                
            # Create actuators from config
            actuators = config_loader.create_actuators_from_config(config, network)
            for actuator in actuators:
                simulator.add_actuator(actuator)
                
            # Create services from config
            services = config_loader.create_services_from_config(config, network)
            for service in services:
                simulator.add_service(service)
                
            # Set simulation parameters from config
            if "simulation" in config:
                sim_config = config["simulation"]
                if "max_time" in sim_config:
                    simulator.max_simulation_time = sim_config["max_time"]
                if "log_level" in sim_config:
                    level_name = sim_config["log_level"]
                    level = getattr(logging, level_name, logging.INFO)
                    logging.getLogger().setLevel(level)
                    simulator.set_logging_level(level)
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            create_default_simulation(simulator)
    else:
        # Create default simulation setup
        create_default_simulation(simulator)
    
    # Add statistics tracking
    simulator.add_observer(lambda event, time, sim: update_statistics(event, time, sim, statistics))
    
    return simulator, statistics

def create_default_simulation(simulator):
    """Create a default simulation setup with some nodes and devices"""
    logger.info("Creating default simulation setup...")
    
    # Create network
    network = Network()
    
    # Create cloud node
    cloud_resources = ComputeResources(cpu_cores=32, cpu_frequency=3.0, ram=65536, storage=1000000)
    cloud_node = Node(
        name="cloud_server",
        node_type=NodeType.CLOUD,
        resources=cloud_resources,
        location=(0.0, 0.0),
        bandwidth=1000.0,
        parent_latency=0.0
    )
    network.add_node(cloud_node)
    simulator.add_node(cloud_node)  # Add to simulator's nodes dictionary
    
    # Create fog node
    fog_resources = ComputeResources(cpu_cores=8, cpu_frequency=2.5, ram=16384, storage=32000)
    fog_node = Node(
        name="fog_node_1",
        node_type=NodeType.FOG,
        resources=fog_resources,
        location=(1.0, 1.0),
        bandwidth=500.0,
        parent_latency=20.0
    )
    network.add_node(fog_node)
    simulator.add_node(fog_node)  # Add to simulator's nodes dictionary
    
    # Create edge node
    edge_resources = ComputeResources(cpu_cores=2, cpu_frequency=1.5, ram=2048, storage=16000)
    edge_node = Node(
        name="edge_node_1",
        node_type=NodeType.EDGE,
        resources=edge_resources,
        location=(1.5, 1.5),
        bandwidth=100.0,
        parent_latency=5.0
    )
    network.add_node(edge_node)
    simulator.add_node(edge_node)  # Add to simulator's nodes dictionary
    
    # Create connections
    network.connect(cloud_node.id, fog_node.id, 1000.0, 20.0)
    network.connect(fog_node.id, edge_node.id, 100.0, 5.0)
    
    # Create hierarchy
    cloud_node.add_child(fog_node)
    fog_node.add_child(edge_node)
    
    # Create sensors
    temp_sensor = Sensor(
        name="temp_sensor_1",
        sensor_type=SensorType.TEMPERATURE,
        location=(1.5, 1.6),
        sampling_rate=0.1,
        data_size=20,
        gateway_id=edge_node.id  # Set the gateway node explicitly
    )
    temp_sensor.attach_to_node(edge_node.id)
    simulator.add_sensor(temp_sensor)
    
    traffic_sensor = Sensor(
        name="traffic_sensor_1",
        sensor_type=SensorType.TRAFFIC,
        location=(1.6, 1.5),
        sampling_rate=1.0,
        data_size=500,
        gateway_id=edge_node.id  # Set the gateway node explicitly
    )
    traffic_sensor.attach_to_node(edge_node.id)
    simulator.add_sensor(traffic_sensor)
    
    # Create actuators
    traffic_light = Actuator(
        name="traffic_light_1",
        actuator_type=ActuatorType.TRAFFIC_LIGHT,
        response_time=0.2,
        location=(1.55, 1.55),
        power_consumption=5.0
    )
    traffic_light.attach_to_node(edge_node.id)
    simulator.add_actuator(traffic_light)
    
    # Create services
    # Traffic monitoring service
    traffic_input_types = {DataType.SENSOR_DATA}
    traffic_output_types = {DataType.COMMAND}
    traffic_service = Service(
        name="traffic_monitoring",
        service_type=ServiceType.TRAFFIC_MANAGEMENT,
        cpu_required=2,
        ram_required=4096,
        storage_required=8000,
        input_types=traffic_input_types,
        output_types=traffic_output_types,
        min_processing_time=0.1,
        max_processing_time=0.5
    )
    traffic_service.deploy(fog_node.id)
    fog_node.deploy_service(traffic_service)
    simulator.add_service(traffic_service)
    
    # Data analytics service
    analytics_input_types = {DataType.SENSOR_DATA, DataType.SYSTEM_DATA}
    analytics_output_types = {DataType.ANALYTICS}
    analytics_service = Service(
        name="data_analytics",
        service_type=ServiceType.DATA_ANALYTICS,
        cpu_required=8,
        ram_required=16384,
        storage_required=32000,
        input_types=analytics_input_types,
        output_types=analytics_output_types,
        min_processing_time=0.5,
        max_processing_time=2.0
    )
    analytics_service.deploy(cloud_node.id)
    cloud_node.deploy_service(analytics_service)
    simulator.add_service(analytics_service)
    
    # Set simulator network
    simulator.network = network
    
    # Make sure services are added to the running_services dictionary of their nodes
    if hasattr(fog_node, 'running_services') and traffic_service.id not in fog_node.running_services:
        fog_node.running_services[traffic_service.id] = traffic_service
    
    if hasattr(cloud_node, 'running_services') and analytics_service.id not in cloud_node.running_services:
        cloud_node.running_services[analytics_service.id] = analytics_service
    
    logger.info("Default simulation setup created")

def update_statistics(event, time, simulator, statistics):
    """Update statistics based on events"""
    # Record event
    statistics.record_event(event.event_type.value)
    
    # Record general metrics
    statistics.record_metric("current_time", time, time)
    statistics.record_metric("events_processed", simulator.metrics["processed_events"], time)
    statistics.record_metric("event_queue_size", len(simulator.event_queue), time)
    
    # Record node resource usage
    for node_id, node in simulator.nodes.items():
        if hasattr(node, 'resources'):
            cpu_usage, ram_usage, storage_usage = node.resources.usage_percentage()
            statistics.record_node_metrics(node_id, {
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "storage_usage": storage_usage
            }, time)
    
    # Record service metrics
    for service_id, service in simulator.services.items():
        statistics.record_service_metrics(service_id, {
            "processed_data_count": service.processed_data_count,
            "processing_time": service.get_average_processing_time()
        }, time)
    
    # Record data transmission metrics for specific event types
    if event.event_type.value == "data_transmission" and "data_id" in event.data:
        data_id = event.data["data_id"]
        if data_id in simulator.data:
            data = simulator.data[data_id]
            if hasattr(event, 'processing_time'):
                statistics.record_data_flow(
                    data_id=data_id,
                    source_id=event.source_id,
                    target_id=event.target_ids[0] if event.target_ids else "unknown",
                    data_size=data.transmission_size if hasattr(data, 'transmission_size') else data.storage_size,
                    transmission_time=event.processing_time,
                    timestamp=time
                )
    
    # Record data processing metrics
    if event.event_type.value == "data_processing" and "data_id" in event.data:
        data_id = event.data["data_id"]
        service_id = event.source_id
        
        if data_id in simulator.data and service_id in simulator.services:
            # Record that this service processed this data item
            statistics.record_metric("data_processed", 1, time)

def run_simulation(simulator, statistics, args):
    """Run the simulation"""
    logger.info(f"Starting simulation with max time {simulator.max_simulation_time}s")
    
    if args.gui:
        # Run with GUI
        logger.info("Launching GUI mode")
        gui = create_gui(simulator)
        gui.run()
    else:
        # Run in command-line mode
        logger.info("Starting command-line simulation")
        simulator.run()
        logger.info(f"Simulation completed at time {simulator.current_time}")
        
        # Print summary
        metrics = simulator.get_metrics()
        logger.info(f"Processed {metrics['processed_events']} events")
        logger.info(f"Generated {metrics['total_data_generated']} data items")
        logger.info(f"Processed {metrics.get('total_data_processed', 0)} data items")
        
        # Export statistics if requested
        if args.output:
            export_statistics(statistics, args.output)

def export_statistics(statistics, output_path):
    """Export statistics to file"""
    if output_path.endswith('.json'):
        if statistics.export_to_json(output_path):
            logger.info(f"Statistics exported to {output_path}")
        else:
            logger.error(f"Failed to export statistics to {output_path}")
    elif output_path.endswith('.csv'):
        # Remove extension for base name
        base_path = output_path[:-4]
        if statistics.export_to_csv(base_path):
            logger.info(f"Statistics exported to CSV files with base name {base_path}")
        else:
            logger.error(f"Failed to export statistics to CSV")
    else:
        # Default to JSON
        json_path = output_path + '.json'
        if statistics.export_to_json(json_path):
            logger.info(f"Statistics exported to {json_path}")
        else:
            logger.error(f"Failed to export statistics to {json_path}")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Set up the simulator
        result = setup_simulator(args)
        if result is None:
            return
            
        simulator, statistics = result
        
        # Run the simulation
        run_simulation(simulator, statistics, args)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())