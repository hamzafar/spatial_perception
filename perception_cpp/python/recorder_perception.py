import csv
import json
from datetime import datetime
from pathlib import Path


class PerceptionRecorder:

    def __init__(
        self,
        output_root="perception_recordings",
        max_frame=350
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        self.output_dir = (
            Path(output_root) /
            f"session_{timestamp}"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.csv_path = (
            self.output_dir /
            "perception.csv"
        )

        self.max_frame = max_frame
        self.frame_count = 0
        self.recording = True

        # Keep complete dashboard payloads in RAM
        # until recording is finished.
        self.frames = []

        print(
            "[PerceptionRecorder] "
            f"Recording up to {self.max_frame} frames"
        )

        print(
            "[PerceptionRecorder] "
            f"Output: {self.output_dir}"
        )

    # -------------------------------------------------

    def record(self, dashboard_data):

        if not self.recording:
            return False

        if not dashboard_data:
            return False

        if self.frame_count >= self.max_frame:
            self.recording = False
            self._save_csv()
            return False

        # -------------------------------------------------
        # Recorder-owned sequential frame ID
        # -------------------------------------------------

        self.frame_count += 1

        frame_id = (
            f"frame_{self.frame_count:06d}"
        )

        # -------------------------------------------------
        # Copy payload so later dashboard mutations
        # cannot modify already-recorded data.
        # -------------------------------------------------

        payload = json.loads(
            json.dumps(dashboard_data)
        )

        payload["frame_id"] = frame_id

        # -------------------------------------------------
        # Do NOT save camera images.
        #
        # Keep all other camera information:
        # boxes, IDs, classes, confidence, etc.
        # -------------------------------------------------

        cameras = payload.get(
            "cameras"
        )

        if isinstance(cameras, dict):

            for camera_data in cameras.values():

                if isinstance(
                    camera_data,
                    dict
                ):

                    camera_data.pop(
                        "image",
                        None
                    )

        # -------------------------------------------------
        # Store complete dashboard payload in RAM
        # -------------------------------------------------

        self.frames.append(
            payload
        )

        # -------------------------------------------------
        # Save CSV ONLY after max_frame
        # -------------------------------------------------

        if self.frame_count >= self.max_frame:

            self.recording = False

            self._save_csv()

            print(
                "[PerceptionRecorder] "
                f"Maximum frames reached: "
                f"{self.max_frame}"
            )

            print(
                "[PerceptionRecorder] "
                f"CSV saved: {self.csv_path}"
            )

        return True

    # -------------------------------------------------

    def _save_csv(self):

        if not self.frames:
            return

        # Collect all payload fields.
        # This also handles a new field appearing
        # during the recording.
        fieldnames = []

        for payload in self.frames:

            for key in payload:

                if key not in fieldnames:
                    fieldnames.append(key)

        # -------------------------------------------------
        # Single CSV write
        # -------------------------------------------------

        with open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for payload in self.frames:

                row = {}

                for key in fieldnames:

                    value = payload.get(
                        key,
                        ""
                    )

                    if isinstance(
                        value,
                        (dict, list, tuple)
                    ):

                        value = json.dumps(
                            value,
                            separators=(",", ":"),
                            ensure_ascii=False
                        )

                    row[key] = value

                writer.writerow(row)

    # -------------------------------------------------

    def close(self):

        if not self.recording:
            return

        self.recording = False

        if self.frames:

            self._save_csv()

            print(
                "[PerceptionRecorder] "
                f"Saved {len(self.frames)} frames"
            )

        else:

            print(
                "[PerceptionRecorder] "
                "No frames recorded"
            )