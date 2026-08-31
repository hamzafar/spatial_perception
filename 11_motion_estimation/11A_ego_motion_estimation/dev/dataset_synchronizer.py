import shutil
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetSynchronizer:

    def __init__(self):
        pass

        self.dataset_path = None

        self.sensors = [
            "front",
            "rear",
            "left",
            "right",
            "lidar",
            "radar",
            "imu",
            "gnss"
        ]

    def load_timestamps(self):
        """
        Load timestamps.csv into a dictionary of NumPy arrays.
        """

        timestamps_df = pd.read_csv(
            self.dataset_path / "timestamps.csv"
        )

        timestamps = {}

        for column in timestamps_df.columns:
            timestamps[column] = timestamps_df[column].to_numpy()

        return timestamps
    
    def find_nearest_timestamp(
        self,
        reference_timestamp,
        sensor_timestamps
    ):
        """
        Return the index of the nearest timestamp.
        """

        return np.argmin(
            np.abs(
                sensor_timestamps -
                reference_timestamp
            )
        )

    def create_sync_dictionary(self, raw_dict):
        """
        Create synchronized frame mapping.
        """

        sync = {}

        reference = raw_dict["front_timestamp"]

        for i, front_timestamp in enumerate(reference):

            sync[i] = {

                "front": i,

                "rear": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["rear_timestamp"]
                ),

                "left": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["left_timestamp"]
                ),

                "right": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["right_timestamp"]
                ),

                "lidar": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["lidar_timestamp"]
                ),

                "imu": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["imu_timestamp"]
                ),

                "gnss": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["gnss_timestamp"]
                ),

                "radar": self.find_nearest_timestamp(
                    front_timestamp,
                    raw_dict["radar_timestamp"]
                )
            }
        return sync

    def print_sync_dictionary(
        self,
        raw_dict,
        sync_dict,
        num_frames=10
    ):

        print("-" * 140)

        for i in range(min(num_frames, len(sync_dict))):

            s = sync_dict[i]

            print(
                f"Front : Frame={raw_dict['front_frame'][s['front']]:6d} "
                f"TS={raw_dict['front_timestamp'][s['front']]:10.6f} | "

                f"Rear : Frame={raw_dict['rear_frame'][s['rear']]:6d} "
                f"TS={raw_dict['rear_timestamp'][s['rear']]:10.6f} | "

                f"Left : Frame={raw_dict['left_frame'][s['left']]:6d} "
                f"TS={raw_dict['left_timestamp'][s['left']]:10.6f} | "

                f"Right: Frame={raw_dict['right_frame'][s['right']]:6d} "
                f"TS={raw_dict['right_timestamp'][s['right']]:10.6f}"
            )

            print(
                f"LiDAR: Frame={raw_dict['lidar_frame'][s['lidar']]:6d} "
                f"TS={raw_dict['lidar_timestamp'][s['lidar']]:10.6f} | "

                f"Radar: Frame={raw_dict['radar_frame'][s['radar']]:6d} "
                f"TS={raw_dict['radar_timestamp'][s['radar']]:10.6f} | "

                f"IMU  : Frame={raw_dict['imu_frame'][s['imu']]:6d} "
                f"TS={raw_dict['imu_timestamp'][s['imu']]:10.6f} | "

                f"GNSS : Frame={raw_dict['gnss_frame'][s['gnss']]:6d} "
                f"TS={raw_dict['gnss_timestamp'][s['gnss']]:10.6f}"
            )

            print("-" * 140)

    def run(self):

        # self.create_output_directories()

        raw_dict = self.load_timestamps()
        sync_dict = self.create_sync_dictionary(raw_dict)

        return sync_dict


def main():

    sync = DatasetSynchronizer()

    sync.run()

if __name__ == "__main__":
    main()