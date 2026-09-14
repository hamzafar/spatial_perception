import numpy as np
import math

class IMUPipeline:

    def __init__(self):

        self.speed_threshold = 0.5          # m/s
        self.acceleration_threshold = 0.3   # m/s²
        self.yaw_rate_threshold = 0.1       # rad/s

    # -------------------------------------------------

    def compute_acceleration(
        self,
        imu_msg
    ):

        return np.array(
            [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z
            ],
            dtype=np.float32
        )

    # -------------------------------------------------

    def compute_yaw_rate(
        self,
        imu_msg
    ):

        return imu_msg.angular_velocity.z

    # -------------------------------------------------

    def compute_heading(
        self,
        imu_msg
    ):

        z = imu_msg.orientation.z
        w = imu_msg.orientation.w

        heading = 2.0 * math.atan2(
            z,
            w
        )

        return heading

    # -------------------------------------------------

    def compute_motion_state(
        self,
        speed,
        acceleration,
        yaw_rate
    ):

        state = []

        if speed < self.speed_threshold:

            state.append("Stationary")

        else:

            state.append("Moving Forward")

        if acceleration[0] > self.acceleration_threshold:

            state.append("Accelerating")

        elif acceleration[0] < -self.acceleration_threshold:

            state.append("Braking")

        # CARLA IMU replay:
        # Positive yaw rate  -> Turning Right
        # Negative yaw rate  -> Turning Left

        if yaw_rate > self.yaw_rate_threshold:

            state.append("Turning Right")

        elif yaw_rate < -self.yaw_rate_threshold:

            state.append("Turning Left")

        return state

    # -------------------------------------------------

    def process(
        self,
        imu_msg,
        speed
    ):

        acceleration = self.compute_acceleration(
            imu_msg
        )

        yaw_rate = self.compute_yaw_rate(
            imu_msg
        )

        heading = self.compute_heading(
            imu_msg
        )

        motion_state = self.compute_motion_state(
            speed,
            acceleration,
            yaw_rate
        )

        return {
            "acceleration": acceleration,
            "yaw_rate": yaw_rate,
            "heading": heading,
            "motion_state": motion_state
        }