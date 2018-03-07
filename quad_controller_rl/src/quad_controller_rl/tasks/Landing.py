"""Landing task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        print("Landing(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 10.0
        max_torque = 10.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        print("Landing(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

    def reset(self):
        # Nothing to reset; just return initial condition
        self.last_timestamp = None
        self.last_position = None
        p = self.target_z + np.random.normal(0.5, 0.1, size=3)  # slight random position around the target
        return Pose(
                position=Point(*p),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose, orientation, velocity only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position
