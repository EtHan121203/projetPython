import json
import csv
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

class Statistics:
    """
    Utility for collecting, analyzing and exporting simulation statistics.
    Tracks metrics over time and provides analysis functions.
    """
    def __init__(self):
        self.metrics = {}  # Metric name -> List of (timestamp, value) tuples
        self.events_by_type = {}  # Event type -> count
        self.network_metrics = {}  # Node ID -> List of resource usage metrics
        self.service_metrics = {}  # Service ID -> List of processing metrics
        self.data_flow_metrics = []  # List of data flow records
        self.logger = logging.getLogger("Statistics")
        self.start_real_time = time.time()
    
    def record_metric(self, name: str, value: Any, timestamp: float) -> None:
        """
        Record a metric value at a specific timestamp
        
        Args:
            name: Name of the metric
            value: Value of the metric
            timestamp: Simulation time when the metric was recorded
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((timestamp, value))
    
    def record_event(self, event_type: str) -> None:
        """
        Record an event occurrence by type
        
        Args:
            event_type: Type of the event
        """
        if event_type not in self.events_by_type:
            self.events_by_type[event_type] = 0
            
        self.events_by_type[event_type] += 1
    
    def record_node_metrics(self, node_id: str, metrics: Dict[str, float], timestamp: float) -> None:
        """
        Record resource usage metrics for a node
        
        Args:
            node_id: ID of the node
            metrics: Dictionary of metric name -> value
            timestamp: Simulation time when metrics were recorded
        """
        if node_id not in self.network_metrics:
            self.network_metrics[node_id] = []
            
        metrics["timestamp"] = timestamp
        self.network_metrics[node_id].append(metrics)
    
    def record_service_metrics(self, service_id: str, metrics: Dict[str, Any], timestamp: float) -> None:
        """
        Record processing metrics for a service
        
        Args:
            service_id: ID of the service
            metrics: Dictionary of metric name -> value
            timestamp: Simulation time when metrics were recorded
        """
        if service_id not in self.service_metrics:
            self.service_metrics[service_id] = []
            
        metrics["timestamp"] = timestamp
        self.service_metrics[service_id].append(metrics)
    
    def record_data_flow(self, data_id: str, source_id: str, target_id: str, 
                         data_size: int, transmission_time: float, timestamp: float) -> None:
        """
        Record a data flow between nodes
        
        Args:
            data_id: ID of the data
            source_id: ID of the source node
            target_id: ID of the target node
            data_size: Size of the data in bytes
            transmission_time: Time taken for transmission in seconds
            timestamp: Simulation time when the flow occurred
        """
        self.data_flow_metrics.append({
            "data_id": data_id,
            "source_id": source_id,
            "target_id": target_id,
            "data_size": data_size,
            "transmission_time": transmission_time,
            "timestamp": timestamp
        })
    
    def get_metric_average(self, name: str) -> Optional[float]:
        """
        Calculate the average value of a metric
        
        Args:
            name: Name of the metric
            
        Returns:
            Average value or None if the metric doesn't exist
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = [value for _, value in self.metrics[name]]
        return sum(values) / len(values)
    
    def get_metric_min(self, name: str) -> Optional[float]:
        """Get the minimum value of a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = [value for _, value in self.metrics[name]]
        return min(values)
    
    def get_metric_max(self, name: str) -> Optional[float]:
        """Get the maximum value of a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = [value for _, value in self.metrics[name]]
        return max(values)
    
    def get_metric_values_over_time(self, name: str) -> List[Tuple[float, Any]]:
        """Get all values of a metric over time"""
        if name not in self.metrics:
            return []
            
        return self.metrics[name]
    
    def get_events_summary(self) -> Dict[str, int]:
        """Get summary of events by type"""
        return self.events_by_type.copy()
    
    def get_node_resource_usage(self, node_id: str) -> List[Dict[str, Any]]:
        """Get resource usage history for a node"""
        if node_id not in self.network_metrics:
            return []
            
        return self.network_metrics[node_id]
    
    def get_service_performance(self, service_id: str) -> List[Dict[str, Any]]:
        """Get performance metrics history for a service"""
        if service_id not in self.service_metrics:
            return []
            
        return self.service_metrics[service_id]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected statistics"""
        return {
            "metrics": self.metrics,
            "events_by_type": self.events_by_type,
            "network_metrics": self.network_metrics,
            "service_metrics": self.service_metrics,
            "data_flow_metrics": self.data_flow_metrics,
            "real_time_elapsed": time.time() - self.start_real_time
        }
    
    def calculate_network_throughput(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Calculate network throughput over time
        
        Returns:
            Dictionary of node ID -> List of (timestamp, throughput) tuples
        """
        # Group data flows by target node and time window
        window_size = 1.0  # 1 second window
        node_throughput = {}
        
        for flow in sorted(self.data_flow_metrics, key=lambda f: f["timestamp"]):
            target = flow["target_id"]
            ts = flow["timestamp"]
            size = flow["data_size"]
            
            # Round timestamp to the nearest window
            window_ts = int(ts / window_size) * window_size
            
            if target not in node_throughput:
                node_throughput[target] = {}
                
            if window_ts not in node_throughput[target]:
                node_throughput[target][window_ts] = 0
                
            node_throughput[target][window_ts] += size
        
        # Convert to bytes per second and format as list of tuples
        result = {}
        for node_id, windows in node_throughput.items():
            result[node_id] = [(ts, size / window_size) for ts, size in windows.items()]
            result[node_id].sort()  # Sort by timestamp
        
        return result
    
    def calculate_average_latency(self) -> float:
        """
        Calculate average network latency across all data flows
        
        Returns:
            Average latency in seconds
        """
        if not self.data_flow_metrics:
            return 0.0
            
        total_latency = sum(flow["transmission_time"] for flow in self.data_flow_metrics)
        return total_latency / len(self.data_flow_metrics)
    
    def calculate_service_utilization(self) -> Dict[str, float]:
        """
        Calculate average utilization percentage for each service
        
        Returns:
            Dictionary of service ID -> utilization percentage
        """
        utilization = {}
        
        for service_id, metrics in self.service_metrics.items():
            # Extract timestamps and processed data counts
            if not metrics:
                utilization[service_id] = 0.0
                continue
                
            # Calculate total simulation time covered
            timestamps = [m["timestamp"] for m in metrics]
            total_time = max(timestamps) - min(timestamps)
            
            if total_time <= 0:
                utilization[service_id] = 0.0
                continue
                
            # Calculate total processing time
            total_processing = sum(m.get("processing_time", 0) for m in metrics)
            
            # Calculate utilization
            utilization[service_id] = min(100.0, (total_processing / total_time) * 100)
        
        return utilization
    
    def calculate_resource_efficiency(self) -> Dict[str, float]:
        """
        Calculate resource efficiency for each node (processed data per resource usage)
        
        Returns:
            Dictionary of node ID -> efficiency score
        """
        efficiency = {}
        
        # Calculate processed data per node
        processed_per_node = {}
        for flow in self.data_flow_metrics:
            target = flow["target_id"]
            if target not in processed_per_node:
                processed_per_node[target] = 0
            processed_per_node[target] += flow["data_size"]
        
        # Calculate average resource usage per node
        for node_id, metrics in self.network_metrics.items():
            if not metrics or node_id not in processed_per_node:
                efficiency[node_id] = 0.0
                continue
                
            # Calculate average CPU and RAM usage
            avg_cpu = sum(m.get("cpu_usage", 0) for m in metrics) / len(metrics)
            avg_ram = sum(m.get("ram_usage", 0) for m in metrics) / len(metrics)
            
            # Resource usage factor (higher usage = lower score)
            usage_factor = (avg_cpu + avg_ram) / 200.0  # Normalize to 0-1 range
            
            # Data processed (higher = better)
            data_processed = processed_per_node.get(node_id, 0)
            
            # Efficiency score: data processed / resource usage
            if usage_factor > 0:
                efficiency[node_id] = data_processed / (usage_factor * 1000)  # Scale for readability
            else:
                efficiency[node_id] = 0.0
        
        return efficiency
    
    def plot_metric_over_time(self, metric_name: str, title: str = None, 
                             xlabel: str = "Time (s)", ylabel: str = None,
                             save_path: str = None, show: bool = True):
        """
        Plot a metric's value over time
        
        Args:
            metric_name: Name of the metric to plot
            title: Plot title (defaults to metric name)
            xlabel: X-axis label
            ylabel: Y-axis label (defaults to metric name)
            save_path: Path to save the plot image
            show: Whether to display the plot
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            self.logger.warning(f"Cannot plot metric '{metric_name}': no data")
            return
        
        # Extract time and values
        times, values = zip(*self.metrics[metric_name])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, values)
        plt.title(title or metric_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric_name)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_events_by_type(self, save_path: str = None, show: bool = True):
        """
        Plot event counts by type
        
        Args:
            save_path: Path to save the plot image
            show: Whether to display the plot
        """
        if not self.events_by_type:
            self.logger.warning("Cannot plot events by type: no data")
            return
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        event_types = list(self.events_by_type.keys())
        counts = list(self.events_by_type.values())
        
        plt.bar(event_types, counts)
        plt.title("Events by Type")
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_resource_usage(self, node_id: str, save_path: str = None, show: bool = True):
        """
        Plot resource usage over time for a specific node
        
        Args:
            node_id: ID of the node
            save_path: Path to save the plot image
            show: Whether to display the plot
        """
        if node_id not in self.network_metrics or not self.network_metrics[node_id]:
            self.logger.warning(f"Cannot plot resource usage for node '{node_id}': no data")
            return
        
        # Extract metrics
        metrics = self.network_metrics[node_id]
        timestamps = [m["timestamp"] for m in metrics]
        cpu_usage = [m.get("cpu_usage", 0) for m in metrics]
        ram_usage = [m.get("ram_usage", 0) for m in metrics]
        storage_usage = [m.get("storage_usage", 0) for m in metrics]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cpu_usage, label="CPU Usage (%)")
        plt.plot(timestamps, ram_usage, label="RAM Usage (%)")
        plt.plot(timestamps, storage_usage, label="Storage Usage (%)")
        plt.title(f"Resource Usage for Node {node_id}")
        plt.xlabel("Time (s)")
        plt.ylabel("Usage (%)")
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_to_json(self, file_path: str) -> bool:
        """
        Export all statistics to a JSON file
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Convert tuples to lists for JSON serialization
            metrics_json = {}
            for name, values in self.metrics.items():
                metrics_json[name] = [[ts, val] for ts, val in values]
            
            data = {
                "metrics": metrics_json,
                "events_by_type": self.events_by_type,
                "network_metrics": self.network_metrics,
                "service_metrics": self.service_metrics,
                "data_flow_metrics": self.data_flow_metrics,
                "analysis": {
                    "average_latency": self.calculate_average_latency(),
                    "service_utilization": self.calculate_service_utilization(),
                    "resource_efficiency": self.calculate_resource_efficiency()
                },
                "real_time_elapsed": time.time() - self.start_real_time
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.logger.info(f"Statistics exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting statistics to {file_path}: {e}")
            return False
    
    def export_to_csv(self, base_file_path: str) -> bool:
        """
        Export statistics to CSV files (one file per metric type)
        
        Args:
            base_file_path: Base path for CSV files (will add suffix for each file)
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Export time series metrics
            metrics_path = f"{base_file_path}_metrics.csv"
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header with all metric names
                header = ["timestamp"] + list(self.metrics.keys())
                writer.writerow(header)
                
                # Get all unique timestamps
                all_timestamps = set()
                for values in self.metrics.values():
                    all_timestamps.update(ts for ts, _ in values)
                all_timestamps = sorted(all_timestamps)
                
                # Create lookup for each metric
                metric_lookup = {}
                for name, values in self.metrics.items():
                    metric_lookup[name] = {ts: val for ts, val in values}
                
                # Write data rows
                for ts in all_timestamps:
                    row = [ts]
                    for name in self.metrics.keys():
                        row.append(metric_lookup[name].get(ts, ""))
                    writer.writerow(row)
            
            # Export node metrics
            node_path = f"{base_file_path}_nodes.csv"
            with open(node_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["node_id", "timestamp", "cpu_usage", "ram_usage", "storage_usage"])
                
                for node_id, metrics in self.network_metrics.items():
                    for m in metrics:
                        writer.writerow([
                            node_id,
                            m.get("timestamp", ""),
                            m.get("cpu_usage", ""),
                            m.get("ram_usage", ""),
                            m.get("storage_usage", "")
                        ])
            
            # Export service metrics
            service_path = f"{base_file_path}_services.csv"
            with open(service_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["service_id", "timestamp", "processed_data_count", "processing_time"])
                
                for service_id, metrics in self.service_metrics.items():
                    for m in metrics:
                        writer.writerow([
                            service_id,
                            m.get("timestamp", ""),
                            m.get("processed_data_count", ""),
                            m.get("processing_time", "")
                        ])
            
            # Export data flow metrics
            flow_path = f"{base_file_path}_flows.csv"
            with open(flow_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["data_id", "source_id", "target_id", "data_size", "transmission_time", "timestamp"])
                
                for flow in self.data_flow_metrics:
                    writer.writerow([
                        flow.get("data_id", ""),
                        flow.get("source_id", ""),
                        flow.get("target_id", ""),
                        flow.get("data_size", ""),
                        flow.get("transmission_time", ""),
                        flow.get("timestamp", "")
                    ])
            
            self.logger.info(f"Statistics exported to CSV files with base {base_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting statistics to CSV: {e}")
            return False

# Helper function to create a Statistics object
def create_statistics():
    return Statistics()