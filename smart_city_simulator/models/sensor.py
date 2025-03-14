# models/node.py
import uuid
from enum import Enum
from typing import Optional, Tuple, Dict
import random
from models.data import DataType

class SensorType(Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    AIR_QUALITY = "air_quality"
    TRAFFIC = "traffic"
    PARKING = "parking"
    NOISE = "noise"
    LIGHT = "light"
    WATER_LEVEL = "water_level"
    ENERGY_CONSUMPTION = "energy_consumption"
    MOTION = "motion"

class Sensor:
    def __init__(self, 
                 name: str,
                 sensor_type: SensorType,
                 location: Tuple[float, float],
                 sampling_rate: float = 1.0,
                 data_size: int = 100,
                 accuracy: float = 0.95,
                 battery_life: Optional[float] = None,
                 range_meters: float = 10.0,
                 power_consumption: float = 0.1,
                 gateway_id: Optional[str] = None,
                 network_latency: float = 50.0  # Default network latency in ms
                ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.sensor_type = sensor_type
        self.location = location
        self.sampling_rate = sampling_rate
        self.generation_frequency = sampling_rate
        self.data_size = data_size
        self.accuracy = accuracy
        self.battery_life = battery_life
        self.range_meters = range_meters
        self.power_consumption = power_consumption
        self.gateway_id = gateway_id
        self.network_latency = network_latency  # Network latency in milliseconds
        self.node_id = None  # Node ID to which the sensor is attached
        self.last_reading = None
        self.is_active = True
        
        # Default values and ranges for different sensor types
        self.value_ranges = {
            SensorType.TEMPERATURE: (-20.0, 50.0, "°C"),  # (min, max, unit)
            SensorType.HUMIDITY: (0.0, 100.0, "%"),
            SensorType.AIR_QUALITY: (0, 500, "AQI"),
            SensorType.TRAFFIC: (0, 100, "vehicles/min"),
            SensorType.PARKING: (0, 1, "occupancy"),
            SensorType.NOISE: (30, 120, "dB"),
            SensorType.LIGHT: (0, 100000, "lux"),
            SensorType.WATER_LEVEL: (0, 10, "meters"),
            SensorType.ENERGY_CONSUMPTION: (0, 10000, "watts"),
            SensorType.MOTION: (0, 1, "detection")
        }
    
    def attach_to_node(self, node_id: str) -> None:
        """Attach the sensor to a node"""
        if not node_id:
            raise ValueError("Cannot attach sensor to None node_id")
        self.node_id = node_id
        # Set the sensor's location to match the node's location if available
        # This helps with future distance calculations
    
    def read(self) -> dict:
        """Generates a sensor reading with random noise based on accuracy"""
        if not self.is_active:
            return None
        
        if self.battery_life is not None:
            # Battery discharge simulation
            self.battery_life -= self.power_consumption / 3600  # In hours
            if self.battery_life <= 0:
                self.is_active = False
                return None
        
        min_val, max_val, unit = self.value_ranges.get(self.sensor_type, (0, 100, "units"))
        
        # Simulating a reading with noise
        base_value = random.uniform(min_val, max_val)
        # Adding noise based on accuracy
        noise_factor = 1.0 - self.accuracy
        noise = random.uniform(-noise_factor * abs(base_value), noise_factor * abs(base_value))
        value = base_value + noise
        
        # Clamp to valid range
        value = max(min_val, min(max_val, value))
        
        timestamp = self.get_current_timestamp()  # Replace with the current time from the simulator
        
        self.last_reading = {
            "sensor_id": self.id,
            "sensor_type": self.sensor_type.value,
            "value": value,
            "unit": unit,
            "timestamp": timestamp,
            "location": self.location
        }
        
        return self.last_reading
    
    def generate_data(self, current_time: float) -> Dict:
        """
        Generates sensor data (adapter for read method)
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with sensor reading and metadata
        """
        reading = self.read()
        
        if reading is None:
            return None
            
        # Add simulation time to the reading
        reading["generated_at"] = current_time
        
        # Add data type and size information that the simulator expects
        reading["data_type"] = DataType.SENSOR_DATA
        reading["size"] = self.data_size
        
        return reading
    
    def get_next_generation_time(self) -> float:
        """
        Calculates the next data generation time based on frequency
        
        Returns:
            Time (in seconds) for the next data generation
        """
        # If frequency is in Hz, next generation is in 1/frequency seconds
        if self.generation_frequency > 0:
            return 1.0 / self.generation_frequency
        return 60.0  # Default to 1 minute if frequency is not set
    
    def calculate_data_rate(self) -> float:
        """Calculate the data rate in bytes/second"""
        return self.sampling_rate * self.data_size
    
    def set_active(self, active: bool) -> None:
        """Activate or deactivate the sensor"""
        self.is_active = active
    
    def get_current_timestamp(self) -> float:
        """
        Returns the current timestamp for the sensor.
        This could be the last reading time or the current time.
        
        Returns:
            Current timestamp in seconds since epoch
        """
        # If you have a last_reading_time property, you could return that
        # Or just return the current time
        import time
        return time.time()
    
    def __str__(self) -> str:
        return f"Sensor {self.name} ({self.sensor_type.value}) at {self.location}"