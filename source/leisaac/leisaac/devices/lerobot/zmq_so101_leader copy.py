import zmq
import json
import threading
import numpy as np
import os
from collections.abc import Callable
from typing import Dict, Tuple
from pynput.keyboard import Listener

from ..device_base import Device
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / np.pi


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180.0


def ticks_to_radians(ticks, joint_name, calibration_data, last_positions, is_first_message=False):
    """Convert servo ticks using calibration data to radians with proper real-to-sim mapping"""
    if ticks is None:
        return last_positions.get(joint_name, 0.0)  # Use last known position
    
    # On first message, if ticks are zero (uninitialized), return initial joint state (0.0 radians)
    if is_first_message and ticks == 0:
        initial_radians = 0.0  # Initial joint states are all 0.0 radians
        last_positions[joint_name] = initial_radians
        return initial_radians
    
    # Get calibration data for this joint
    if joint_name not in calibration_data:
        return 0.0
    
    cal = calibration_data[joint_name]
    range_min = cal["range_min"]
    range_max = cal["range_max"]
    homing_offset = cal.get("homing_offset", 0)
    
    # Apply homing offset to get the actual position
    adjusted_ticks = ticks + homing_offset
    
    # Clamp ticks to valid range
    adjusted_ticks = max(range_min, min(range_max, adjusted_ticks))
    
    # The key insight: map the calibrated tick range to the robot's actual joint range
    # For SO101, based on the motor limits, most joints have ~200 degree range (-100 to +100)
    # The gripper has 0 to 100 degree range
    
    # Normalize to 0-1 based on the joint's specific tick range
    normalized = (adjusted_ticks - range_min) / (range_max - range_min)
    
    # Map to appropriate radian range based on joint type
    if joint_name == 'gripper':
        # Gripper: 0 to 100 degrees -> 0 to ~1.75 radians
        radians = normalized * degrees_to_radians(100.0)
    else:
        # Other joints: -100 to 100 degrees -> -1.75 to 1.75 radians  
        radians = (normalized - 0.5) * degrees_to_radians(200.0)
    
    # Update last known position
    last_positions[joint_name] = radians
    
    return radians


class ZMQSO101Leader(Device):
    """A SO101 Leader device that receives joint states via ZMQ.
    
    This device listens for absolute joint state messages over ZMQ instead of
    reading directly from hardware. The joint states should be sent as a JSON
    dictionary with keys matching the joint names.
    """

    def __init__(self, env, zmq_port: int = 5555, zmq_host: str = "localhost", input_in_radians: bool = False, input_in_ticks: bool = True):
        super().__init__(env)
        self.zmq_port = zmq_port
        self.zmq_host = zmq_host
        self.input_in_radians = input_in_radians
        self.input_in_ticks = input_in_ticks
        
        # Load calibration data for tick conversion
        calibration_path = os.path.join(os.path.dirname(__file__), "calibration.json")
        with open(calibration_path, 'r') as f:
            self.calibration = json.load(f)
        
        # Initialize joint state storage - internal state that gets updated by ZMQ
        # Will be initialized with first received values (converted to degrees)
        self._joint_state = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # Flag to track if we've received the first message
        self._initialized = False
        
        # Last known positions for handling None values (use joint names as keys)
        self.last_positions = {name: 0.0 for name in self._joint_state.keys()}
        
        # Joint name to index mapping
        self.joint_to_idx = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_roll": 4,
            "gripper": 5,
        }
        
        # Motor limits from the follower
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{zmq_host}:{zmq_port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout for non-blocking receive
        
        # Start background thread for receiving joint states
        self._running = True
        self._receiver_thread = threading.Thread(target=self._receive_joint_states)
        self._receiver_thread.daemon = True
        self._receiver_thread.start()
        
        # Wait for first message to initialize joint state
        print("Waiting for first ZMQ message to initialize joint state...")
        while not self._initialized and self._running:
            import time
            time.sleep(0.01)
        
        # Control flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        
        # Keyboard listener for control commands
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()
        
        print(f"ZMQSO101Leader connected to tcp://{zmq_host}:{zmq_port}")

    def __str__(self) -> str:
        """Returns: A string containing the information of ZMQ SO101 leader."""
        msg = "ZMQ SO101-Leader device for SE(3) control.\n"
        msg += f"\tListening on: tcp://{self.zmq_host}:{self.zmq_port}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tSupported JSON message formats:\n"
        msg += "\t  1. Direct format: {\"shoulder_pan\": float, \"shoulder_lift\": float, ...}\n"
        msg += "\t  2. ROS JointState format: {\"joint_names\": [...], \"joint_positions\": [...]}\n"
        if self.input_in_ticks:
            unit = "servo ticks (0-4095)"
        elif self.input_in_radians:
            unit = "radians"
        else:
            unit = "degrees"
        msg += f"\t  Values should be in {unit}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tControls:\n"
        msg += "\t  b - start control\n"
        msg += "\t  r - reset simulation (task failed)\n"
        msg += "\t  n - reset simulation (task success)\n"
        return msg

    def __del__(self):
        """Cleanup ZMQ resources."""
        self._running = False
        if hasattr(self, '_receiver_thread'):
            self._receiver_thread.join(timeout=1.0)
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def _display_controls(self):
        """Method to pretty print controls."""
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start control")
        print_command("r", "reset simulation and set task success to False")
        print_command("n", "reset simulation and set task success to True")
        print_command("stream joint states via ZMQ", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    def _receive_joint_states(self):
        """Background thread to receive joint states from ZMQ and update internal state."""
        while self._running:
            try:
                # Try to receive a message with timeout
                message = self.socket.recv_string()
                data = json.loads(message)
                
                # Check if this is ROS-style format (from ros_zmq_bridge.py)
                if "joint_names" in data and "joint_positions" in data:
                    # Handle ROS JointState format
                    joint_names = data["joint_names"]
                    joint_positions = data["joint_positions"]
                    
                    # Map joint names to positions
                    for i, name in enumerate(joint_names):
                        if name in self._joint_state and i < len(joint_positions):
                            value = joint_positions[i]
                            
                            # Convert based on input format
                            if self.input_in_ticks:
                                # Convert from ticks to radians first
                                value = ticks_to_radians(value, name, self.calibration, self.last_positions, not self._initialized)
                                # Then convert to degrees for internal storage
                                value = radians_to_degrees(value)
                            elif self.input_in_radians:
                                # Convert from radians to degrees
                                value = radians_to_degrees(float(value))
                            else:
                                # Already in degrees
                                value = float(value)
                            
                            # Ensure the value is within motor limits (in degrees)
                            if name in self._motor_limits:
                                min_val, max_val = self._motor_limits[name]
                                value = np.clip(value, min_val, max_val)
                            self._joint_state[name] = value
                
                else:
                    # Handle direct key-value format
                    for joint_name in self._joint_state.keys():
                        if joint_name in data:
                            value = data[joint_name]
                            
                            # Convert based on input format
                            if self.input_in_ticks:
                                # Convert from ticks to radians first
                                value = ticks_to_radians(value, joint_name, self.calibration, self.last_positions, not self._initialized)
                                # Then convert to degrees for internal storage
                                value = radians_to_degrees(value)
                            elif self.input_in_radians:
                                # Convert from radians to degrees
                                value = radians_to_degrees(float(value))
                            else:
                                # Already in degrees
                                value = float(value)
                            
                            # Ensure the value is within motor limits (in degrees)
                            if joint_name in self._motor_limits:
                                min_val, max_val = self._motor_limits[joint_name]
                                value = np.clip(value, min_val, max_val)
                            self._joint_state[joint_name] = value
                
                # Mark as initialized after processing first message
                if not self._initialized:
                    self._initialized = True
                    print(f"Joint state initialized with first ZMQ message: {self._joint_state}")
                        
            except zmq.Again:
                # Timeout occurred, continue
                continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON message: {e}")
            except Exception as e:
                if self._running:
                    print(f"Error receiving joint states: {e}")

    def on_press(self, key):
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key: key that was pressed
        """
        try:
            if key.char == 'b':
                self._started = True
                self._reset_state = False
                print("Control started")
            elif key.char == 'r':
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
                print("Reset (task failed)")
            elif key.char == 'n':
                self._started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
                print("Reset (task success)")
        except AttributeError:
            # Key doesn't have char attribute (e.g., special keys)
            pass

    def get_device_state(self):
        """Return the current internal joint state (updated by ZMQ messages)."""
        if not self._initialized:
            # Return zeros if not initialized yet
            return {name: 0.0 for name in self._joint_state.keys()}
        
        # All values should be properly initialized after first message
        return self._joint_state.copy()

    def input2action(self):
        """Convert the current joint state to an action dictionary."""
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self._started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self._started
        ac_dict['so101_leader'] = True  # Use same flag as regular SO101Leader for compatibility
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = self._motor_limits
        return ac_dict

    def reset(self):
        """Reset the device state."""
        # Reset joint states to zero
        for key in self._joint_state:
            self._joint_state[key] = 0.0

    def add_callback(self, key: str, func: Callable):
        """Add additional callback functions."""
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._motor_limits