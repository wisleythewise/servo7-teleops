import os
import json
import threading
from typing import Dict, Tuple
from collections.abc import Callable

import zmq
import numpy as np
from pynput.keyboard import Listener

from ..device_base import Device
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS


# -------------------------------
# Unit helpers
# -------------------------------

def radians_to_degrees(radians: float) -> float:
    return radians * 180.0 / np.pi


def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


# -------------------------------
# ZMQ SO101 Leader Device
# -------------------------------


"""
python scripts/environments/teleoperation/teleop_se3_agent.py --task=LeIsaac-SO101-PickOrange-v0 --teleop_device=zmq-so101leader --num_envs=1 --device=cpu --enable_cameras --record --dataset_file=./datasets/dataset.hdf5

"""

class ZMQSO101Leader(Device):
    """
    A SO101 Leader device that receives absolute joint states over ZMQ.

    Internal representation: **radians** (always).
    External accessors: return **degrees** by default (see get_device_state, input2action).

    Input formats supported:
      1) Direct dict:
         {
           "shoulder_pan": float,
           "shoulder_lift": float,
           "elbow_flex": float,
           "wrist_flex": float,
           "wrist_roll": float,
           "gripper": float
         }

      2) ROS JointState-like:
         {
           "joint_names": [...],
           "joint_positions": [...]
         }

    Input units depend on flags:
      - input_in_ticks=True   -> values are ticks, converted (via calibration) to radians
      - input_in_radians=True -> values are radians (stored directly)
      - else                  -> values are degrees (converted to radians)

    Initialization:
      - The very first successfully parsed values define the initial joint states.
      - No special zeroing is performed unless the first message actually contains zeros.
    """

    def __init__(
        self,
        env,
        zmq_port: int = 5555,
        zmq_host: str = "localhost",
        input_in_radians: bool = False,
        input_in_ticks: bool = True,
    ):
        super().__init__(env)
        self.zmq_port = zmq_port
        self.zmq_host = zmq_host
        self.input_in_radians = input_in_radians
        self.input_in_ticks = input_in_ticks

        # Load calibration for tick conversion
        calibration_path = os.path.join(os.path.dirname(__file__), "calibration.json")
        with open(calibration_path, "r") as f:
            self._calibration: Dict[str, Dict[str, float]] = json.load(f)

        # Internal joint state (always radians)
        self._joint_state_rad: Dict[str, float] = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        # Last known positions (radians) for resilience when a field is missing/None
        self._last_positions_rad: Dict[str, float] = {k: 0.0 for k in self._joint_state_rad.keys()}

        self._initialized = False  # flips True after first successfully parsed message

        self.joint_to_idx = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_roll": 4,
            "gripper": 5,
        }

        # Import degree limits and convert once to radians for internal clipping
        self._motor_limits_deg: Dict[str, Tuple[float, float]] = SO101_FOLLOWER_MOTOR_LIMITS
        self._motor_limits_rad: Dict[str, Tuple[float, float]] = {
            j: (degrees_to_radians(v[0]), degrees_to_radians(v[1]))
            for j, v in self._motor_limits_deg.items()
        }

        # ZMQ subscriber
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{zmq_host}:{zmq_port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10 ms timeout

        # Background receiver
        self._running = True
        self._receiver_thread = threading.Thread(target=self._receive_joint_states, daemon=True)
        self._receiver_thread.start()

        # Wait for first successfully parsed message to initialize
        print("Waiting for first ZMQ message to initialize joint state...")
        while not self._initialized and self._running:
            import time
            time.sleep(0.01)

        # Control flags and callbacks
        self._started = False
        self._reset_state = False
        self._gripper_halfway = False  # Flag for gripper halfway position
        self._additional_callbacks: Dict[str, Callable] = {}

        # Keyboard listener
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()

        print(f"ZMQSO101Leader connected to tcp://{zmq_host}:{zmq_port}")

    # ---------------
    # Lifecycle
    # ---------------

    def __del__(self):
        """Cleanup ZMQ resources and stop keyboard listener."""
        self._running = False
        try:
            if hasattr(self, "_receiver_thread") and self._receiver_thread.is_alive():
                self._receiver_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if hasattr(self, "listener"):
                self.listener.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
        except Exception:
            pass

    # ---------------
    # UI / Help
    # ---------------

    def __str__(self) -> str:
        msg = "ZMQ SO101-Leader device for SE(3) control.\n"
        msg += f"\tListening on: tcp://{self.zmq_host}:{self.zmq_port}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tSupported JSON formats:\n"
        msg += "\t  1) Direct: {\"shoulder_pan\": float, ...}\n"
        msg += "\t  2) ROS JointState: {\"joint_names\": [...], \"joint_positions\": [...]}\n"
        if self.input_in_ticks:
            unit = "servo ticks (0-4095)"
        elif self.input_in_radians:
            unit = "radians"
        else:
            unit = "degrees"
        msg += f"\t  Inputs expected in {unit}; internally stored in radians. Public APIs return degrees by default.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tControls:\n"
        msg += "\t  b - start control\n"
        msg += "\t  r - reset simulation (task failed)\n"
        msg += "\t  n - reset simulation (task success)\n"
        return msg

    def _display_controls(self):
        def print_command(char, info):
            char += " " * (30 - len(char))
            print(f"{char}\t{info}")
        print("")
        print_command("b", "start control")
        print_command("r", "reset simulation and set task success to False")
        print_command("n", "reset simulation and set task success to True")
        print_command("stream joint states via ZMQ", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    # ---------------
    # Conversion (member) — ticks -> radians
    # ---------------

    def _ticks_to_radians(self, ticks, joint_name: str) -> float:
        """
        Convert servo ticks to radians using per-joint calibration.

        Behavior:
          - If ticks is None: return last known radians (default 0.0 if unseen).
          - Otherwise: apply calibration mapping to radians and return it.

        Notes:
          - No special-casing for the first message: whatever arrives first becomes
            the initial state (after conversion).
        """
        if ticks is None:
            return self._last_positions_rad.get(joint_name, 0.0)

        cal = self._calibration.get(joint_name)
        if not cal:
            # Unknown joint: keep last, or 0.0 if unseen.
            return self._last_positions_rad.get(joint_name, 0.0)

        range_min = cal["range_min"]
        range_max = cal["range_max"]
        homing_offset = cal.get("homing_offset", 0)

        denom = (range_max - range_min)
        if denom == 0:
            # Bad calibration: keep last.
            return self._last_positions_rad.get(joint_name, 0.0)

        # Apply homing offset and clamp to calibrated range.
        adjusted_ticks = ticks + homing_offset
        adjusted_ticks = max(range_min, min(range_max, adjusted_ticks))

        # Normalize [range_min, range_max] -> [0, 1]
        normalized = (adjusted_ticks - range_min) / denom

        # Map normalized ticks to joint's mechanical span in radians.
        # For SO101:
        # - gripper: 0..100 deg  -> 0..~1.745 rad
        # - others : -100..+100  -> ~-1.745..+1.745 rad
        if joint_name == "gripper":
            radians_val = normalized * degrees_to_radians(100.0)
        elif joint_name == "wrist":
            # Flip axis for shoulder_pan
            radians_val += np.pi 
        else:
            radians_val = (normalized - 0.5) * degrees_to_radians(200.0)

        # Persist last known (radians)
        self._last_positions_rad[joint_name] = radians_val
        return radians_val

    # ---------------
    # Input handling (ZMQ thread)
    # ---------------

    def _receive_joint_states(self):
        """Receive messages and update internal joint state in **radians**."""
        while self._running:
            try:
                msg = self.socket.recv_string()
                data = json.loads(msg)

                updated_any = False

                if "joint_names" in data and "joint_positions" in data:
                    joint_names = data["joint_names"]
                    joint_positions = data["joint_positions"]

                    for i, name in enumerate(joint_names):
                        if name not in self._joint_state_rad or i >= len(joint_positions):
                            continue

                        raw_val = joint_positions[i]

                        if self.input_in_ticks:
                            val_rad = self._ticks_to_radians(raw_val, name)
                        elif self.input_in_radians:
                            val_rad = float(raw_val)
                        else:
                            # degrees input -> store radians
                            val_rad = degrees_to_radians(float(raw_val))

                        # Clip in radians
                        if name in self._motor_limits_rad:
                            lo, hi = self._motor_limits_rad[name]
                            val_rad = float(np.clip(val_rad, lo, hi))

                        self._joint_state_rad[name] = val_rad
                        updated_any = True

                else:
                    # Direct dict format
                    for name in self._joint_state_rad.keys():
                        if name not in data:
                            continue

                        raw_val = data[name]

                        if self.input_in_ticks:
                            val_rad = self._ticks_to_radians(raw_val, name)
                        elif self.input_in_radians:
                            val_rad = float(raw_val)
                        else:
                            val_rad = degrees_to_radians(float(raw_val))

                        if name in self._motor_limits_rad:
                            lo, hi = self._motor_limits_rad[name]
                            val_rad = float(np.clip(val_rad, lo, hi))

                        self._joint_state_rad[name] = val_rad
                        updated_any = True

                # First successfully parsed message defines the initial state.
                if updated_any and not self._initialized:
                    self._initialized = True
                    print(
                        "Joint state initialized (degrees): "
                        f"{ {k: radians_to_degrees(v) for k, v in self._joint_state_rad.items()} }"
                    )

            except zmq.Again:
                # Non-blocking timeout; keep polling
                continue
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON message: {e}")
            except Exception as e:
                if self._running:
                    print(f"Error receiving joint states: {e}")

    # ---------------
    # Keyboard controls
    # ---------------

    def on_press(self, key):
        pass

    def on_release(self, key):
        try:
            if key.char == "b":
                self._started = True
                self._reset_state = False
                print("Control started")
            elif key.char == "r":
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
                print("Reset (task failed)")
            elif key.char == "n":
                self._started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
                print("Reset (task success)")
            elif key.char == "p":
                # Toggle gripper halfway position
                self._gripper_halfway = not self._gripper_halfway
                print(f"Gripper halfway mode: {'ON' if self._gripper_halfway else 'OFF'}")
        except AttributeError:
            # Non-printable/special keys
            pass

    # ---------------
    # Public API (external exposure -> degrees by default)
    # ---------------

    def get_device_state(self, as_degrees: bool = True) -> Dict[str, float]:
        """
        Returns the current joint state.
          - Internally stored in radians.
          - Returns degrees by default (as_degrees=True).
        If not initialized yet, returns zeros.
        """
        if not self._initialized:
            if as_degrees:
                return {name: 0.0 for name in self._joint_state_rad.keys()}
            else:
                return self._joint_state_rad.copy()

        # Get the current state
        state = self._joint_state_rad.copy()
        
        # Override gripper position if halfway mode is active
        if self._gripper_halfway:
            # Set gripper to halfway position (pi/4 radians = 45 degrees)
            state["gripper"] = np.pi / 4
        else: 
            state["gripper"] = np.pi / 2 
        
        if as_degrees:
            return {name: radians_to_degrees(val) for name, val in state.items()}
        else:
            return state

    def input2action(self, output_degrees: bool = True) -> Dict:
        """
        Returns the action dict expected by downstream consumers.
        By default, exposes **degrees** externally (output_degrees=True).

        Keys:
          - "reset": bool
          - "started": bool
          - "so101_leader": True
          - "joint_state": dict of joint -> value (degrees by default)
          - "motor_limits": dict of joint -> (min, max) (degrees by default)
        """
        reset = self._reset_state
        ac_dict = {
            "reset": reset,
            "started": self._started,
            "so101_leader": True,
        }
        if reset:
            self._reset_state = False
            return ac_dict

        if output_degrees:
            ac_dict["joint_state"] = self.get_device_state(as_degrees=True)
            ac_dict["motor_limits"] = self._motor_limits_deg
        else:
            # Advanced use: expose radians
            ac_dict["joint_state"] = self.get_device_state(as_degrees=False)
            ac_dict["motor_limits"] = self._motor_limits_rad

        return ac_dict

    def reset(self):
        """Reset the device state to zeros (radians). Does not alter initialization flag."""
        for k in self._joint_state_rad:
            self._joint_state_rad[k] = 0.0

    def add_callback(self, key: str, func: Callable):
        """Register additional callbacks, e.g., for 'R'/'N' reset hooks."""
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, v: bool):
        self._reset_state = v

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        """Motor limits (degrees by default for external consumers)."""
        return self._motor_limits_deg
