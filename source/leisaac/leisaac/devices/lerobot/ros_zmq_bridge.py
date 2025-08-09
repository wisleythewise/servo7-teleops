#!/usr/bin/env python3
"""
ROS to ZMQ bridge for SO100 arm teleoperation.

This script runs outside the conda environment and bridges ROS joint_states messages to ZMQ.
Run this with system Python that has ROS installed:

    source /opt/ros/humble/setup.bash
    python3 ros_zmq_bridge.py

Then run IsaacLab with the ZMQ device in your conda environment.
"""

import json
import time


def main():
    """Main bridge loop."""
    try:
        # Import ROS (system Python with ROS installed)
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        print("[INFO] ROS 2 imported successfully")
    except ImportError:
        try:
            # Fallback to ROS 1
            import rospy
            from sensor_msgs.msg import JointState
            print("[INFO] ROS 1 imported successfully")
        except ImportError:
            print("[ERROR] Neither ROS 1 nor ROS 2 available")
            return

    try:
        # Import ZMQ
        import zmq
        print("[INFO] ZMQ imported successfully")
    except ImportError:
        print("[ERROR] ZMQ not available. Install with: pip install pyzmq")
        return

    # Initialize ZMQ publisher
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")
    print("[INFO] ZMQ publisher bound to port 5555")

    # Joint name mapping from ROS names to internal names
    joint_mapping = {
        "Rotation": "shoulder_pan",
        "Pitch": "shoulder_lift",
        "Elbow": "elbow_flex",
        "Wrist_Pitch": "wrist_flex",
        "Wrist_Roll": "wrist_roll",
        "Jaw": "gripper"
    }

    def joint_state_callback(msg):
        """Convert ROS JointState to ZMQ JSON message."""
        # Map joint names if they exist in the mapping, otherwise keep original
        mapped_names = []
        for name in msg.name:
            mapped_name = joint_mapping.get(name, name)
            mapped_names.append(mapped_name)
        
        # Create JSON message with mapped joint names and positions
        zmq_msg = {
            "timestamp": time.time(),
            "frame_id": msg.header.frame_id if msg.header.frame_id else '',
            "joint_names": mapped_names,
            "joint_positions": list(msg.position),
            "joint_velocities": list(msg.velocity) if msg.velocity else [],
            "joint_efforts": list(msg.effort) if msg.effort else []
        }
        
        # Send via ZMQ
        zmq_msg_str = json.dumps(zmq_msg)
        publisher.send_string(zmq_msg_str)
        print(f"[DEBUG] Bridged joint_states: {len(msg.name)} joints")
        print(f"[DEBUG] ZMQ message: {zmq_msg_str}")

    # ROS 2 setup
    if 'rclpy' in locals():
        rclpy.init()
        
        class BridgeNode(Node):
            def __init__(self):
                super().__init__('ros_zmq_bridge')
                self.subscription = self.create_subscription(
                    JointState,
                    'joint_states',
                    joint_state_callback,
                    1
                )
                
        node = BridgeNode()
        print("[INFO] ROS 2 bridge node created, subscribing to 'joint_states'")
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            print("[INFO] Shutting down bridge...")
        finally:
            node.destroy_node()
            rclpy.shutdown()
            publisher.close()
            context.term()
    
    # ROS 1 setup
    elif 'rospy' in locals():
        rospy.init_node('ros_zmq_bridge', anonymous=True)
        rospy.Subscriber('joint_states', JointState, joint_state_callback)
        print("[INFO] ROS 1 bridge node created, subscribing to 'joint_states'")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("[INFO] Shutting down bridge...")
        finally:
            publisher.close()
            context.term()


if __name__ == '__main__':
    main()