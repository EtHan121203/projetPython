# models/node.py
import uuid
from enum import Enum
from typing import Optional, Tuple

class ActuatorType(Enum):
    TRAFFIC_LIGHT = "traffic_light"
    STREET_LIGHT = "street_light"
    IRRIGATION = "irrigation"
    HVAC = "hvac"
    BARRIER = "barrier"
    DISPLAY = "display"
    ALERT = "alert"
    ENERGY_CONTROL = "energy_control"

class ActuatorState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FAILED = "failed"

class Actuator:
    def __init__(self, actuator_type: ActuatorType, node_id: Optional[str] = None, name: Optional[str] = None, response_time: Optional[float] = None, location: Optional[Tuple[float, float]] = None):
        self.id = str(uuid.uuid4())
        self.actuator_type = actuator_type
        self.node_id = node_id
        self.name = name
        self.response_time = response_time
        self.location = location
        self.state = ActuatorState.IDLE
        self.current_value = None
        self.actions = {
            ActuatorType.TRAFFIC_LIGHT: ["red", "yellow", "green"],
            ActuatorType.STREET_LIGHT: ["off", "dim", "bright"],
            ActuatorType.IRRIGATION: ["off", "low", "medium", "high"],
            ActuatorType.HVAC: ["off", "heat", "cool", "fan"],
            ActuatorType.BARRIER: ["open", "closed"],
            ActuatorType.DISPLAY: ["off", "on", "message"],
            ActuatorType.ALERT: ["off", "low", "medium", "high"],
            ActuatorType.ENERGY_CONTROL: ["off", "on", "eco"]
        }

    def attach_to_node(self, node_id: str) -> None:
        """Attach the actuator to a node"""

        self.node_id = node_id

    def is_action_valid(self, action: str) -> bool:
        """Check if an action is valid for this type of actuator"""

        return action in self.actions.get(self.actuator_type, [])