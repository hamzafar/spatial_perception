import numpy as np
import pymap3d as pm


class GNSSPipeline:

    def __init__(self):

        self.reference_set = False

        self.lat0 = None
        self.lon0 = None
        self.alt0 = None

        self.previous_position = None
        self.previous_timestamp = None

    # -------------------------------------------------

    def set_reference(
        self,
        latitude,
        longitude,
        altitude
    ):

        self.lat0 = latitude
        self.lon0 = longitude
        self.alt0 = altitude

        self.reference_set = True

    # -------------------------------------------------

    def compute_position(
        self,
        latitude,
        longitude,
        altitude
    ):

        if not self.reference_set:

            self.set_reference(
                latitude,
                longitude,
                altitude
            )

        east, north, up = pm.geodetic2enu(
            latitude,
            longitude,
            altitude,
            self.lat0,
            self.lon0,
            self.alt0
        )

        return np.array(
            [
                east,
                north,
                up
            ],
            dtype=np.float32
        )

    # -------------------------------------------------

    def compute_speed(
        self,
        position,
        timestamp
    ):

        if self.previous_position is None:

            self.previous_position = position
            self.previous_timestamp = timestamp

            return 0.0

        dt = timestamp - self.previous_timestamp

        if dt <= 0:

            return 0.0

        distance = np.linalg.norm(
            position - self.previous_position
        )

        speed = distance / dt

        self.previous_position = position
        self.previous_timestamp = timestamp

        return speed

    # -------------------------------------------------

    def process(
        self,
        latitude,
        longitude,
        altitude,
        timestamp
    ):

        position = self.compute_position(
            latitude,
            longitude,
            altitude
        )

        speed = self.compute_speed(
            position,
            timestamp
        )

        return {
            "position": position,
            "speed": speed
        }