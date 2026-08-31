import cv2
import numpy as np


class DisplayPipeline:
    """
    Layout:

        ┌───────────────────────────────┐
        │          Front Camera          │
        │  (telemetry text overlaid)     │
        ├────────────────────┬───────────┤
        │  Ego Trajectory     │  Compass  │
        └────────────────────┴───────────┘

    Camera sets the total canvas width. The bottom row is split into a wide
    trajectory panel and a narrower compass panel, sized by TRAJECTORY_RATIO.
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        bottom_height: int = 260,
        trajectory_ratio: float = 0.7,
    ):
        self.window_name = "Phase 11.1 - Ego Vehicle Motion"

        self.image_width = image_width
        self.image_height = image_height

        self.bottom_height = bottom_height
        self.trajectory_width = int(image_width * trajectory_ratio)
        self.compass_width = image_width - self.trajectory_width

        self.status_bar_height = 60

        self.canvas_width = image_width
        self.canvas_height = image_height + bottom_height + self.status_bar_height

        self.trajectory = []
        self.trajectory_scale = 10.0     # pixels per metre in the trajectory panel
        self.trajectory_max_points = 500  # cap history so the panel doesn't slow down over long runs

        # blink state for the status bar indicators
        self.blink_half_period = 10  # frames per on/off phase -- lower = faster blink
        # per-indicator activation tracking, so each LED's blink phase locks
        # to the exact frame it turns true (instant flash, no lag waiting on
        # a shared global clock)
        self.frame_count = 0
        self.indicator_state = {}

        # (motion_state label -> indicator color, BGR). Matched case-insensitively
        # against imu["motion_state"]. Edit labels here if your motion_state
        # strings differ (e.g. "Accel" instead of "Accelerating").
        self.status_indicators = [
            ("Accelerating",  (0, 255, 0)),    # green
            ("Turning Left",  (0, 165, 255)),  # orange
            ("Turning Right", (0, 165, 255)),  # orange
            ("Braking",       (0, 0, 255)),    # red
        ]

    # -------------------------------------------------

    def initialize_display(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)

    # -------------------------------------------------
    # CAMERA PANEL (top) - image with a translucent HUD box of telemetry text
    # -------------------------------------------------

    def draw_camera_panel(self, image, gnss, imu):
        if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
            image = cv2.resize(image, (self.image_width, self.image_height))
        else:
            image = image.copy()

        position = gnss["position"]
        speed = gnss["speed"]
        heading_deg = np.degrees(imu["heading"])
        # motion = ", ".join(imu["motion_state"])
        motion = "Moving Forward" if speed > 0 else "Stationary"

        lines = [
            f"Position : ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m",
            f"Speed    : {speed:.2f} m/s",
            f"Heading  : {heading_deg:.2f} deg",
            f"Motion   : {motion}",
        ]

        pad = 10
        line_h = 26
        box_h = pad * 2 + line_h * len(lines)
        box_w = 360

        # translucent HUD box so the video underneath stays visible
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        y = pad + 18
        for text in lines:
            cv2.putText(
                image, text, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )
            y += line_h

        cv2.putText(
            image, "Front Camera", (self.image_width - 170, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return image

    # -------------------------------------------------
    # TRAJECTORY PANEL (bottom-left)
    # -------------------------------------------------

    def draw_trajectory_panel(self, gnss, imu):
        """Ego-relative (heading-up) trajectory: the vehicle stays fixed at
        the panel center facing 'up', and the recorded path is rotated by
        the current heading each frame so it moves/turns around the vehicle
        instead of staying fixed to true-north/world axes."""
        w, h = self.trajectory_width, self.bottom_height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        position = gnss["position"]
        heading = imu["heading"]
        self.trajectory.append((position[0], position[1]))
        if len(self.trajectory) > self.trajectory_max_points:
            self.trajectory = self.trajectory[-self.trajectory_max_points:]

        cx, cy = w // 2, h // 2
        ego_x, ego_y = self.trajectory[-1]

        sin_h, cos_h = np.sin(heading), np.cos(heading)

        for wx, wy in self.trajectory:
            dx = wx - ego_x
            dy = wy - ego_y
            # rotate world offset into the vehicle's body frame:
            # forward = component along current heading, right = perpendicular
            forward = dx * sin_h + dy * cos_h
            right = dx * cos_h - dy * sin_h

            px = int(cx + right * self.trajectory_scale)
            py = int(cy - forward * self.trajectory_scale)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 2, (0, 200, 0), -1)

        # ego is always fixed at panel center, facing up
        cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
        cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 18), (0, 0, 255), 2, tipLength=0.4)

        cv2.putText(
            canvas, "Ego Trajectory", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return canvas

    # -------------------------------------------------
    # COMPASS PANEL (bottom-right)
    # -------------------------------------------------

    def draw_compass_panel(self, imu):
        """Heading-up compass: the ego arrow is fixed pointing straight up
        (forward); N/E/S/W rotate around it based on current heading, so the
        display always shows bearings relative to the vehicle, not true north."""
        w, h = self.compass_width, self.bottom_height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        cx, cy = w // 2, h // 2 + 10
        radius = min(w, h) // 2 - 30

        cv2.circle(canvas, (cx, cy), radius, (90, 90, 90), 1, cv2.LINE_AA)

        heading = imu["heading"]

        # ego is always fixed pointing straight up
        tip_x, tip_y = cx, cy - radius
        cv2.arrowedLine(
            canvas, (cx, cy), (tip_x, tip_y),
            (0, 255, 0), 2, tipLength=0.25,
        )

        # N/E/S/W positioned relative to current heading, not true north
        for label, true_angle in (("N", 0), ("E", np.pi / 2), ("S", np.pi), ("W", -np.pi / 2)):
            rel_angle = true_angle - heading
            lx = int(cx + (radius + 15) * np.sin(rel_angle))
            ly = int(cy - (radius + 15) * np.cos(rel_angle))
            cv2.putText(
                canvas, label, (lx - 6, ly + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
            )

        cv2.putText(
            canvas, "Ego", (cx - 15, cy + radius + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "Compass", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return canvas

    # -------------------------------------------------
    # STATUS BAR (bottom strip) - blinking indicators
    # -------------------------------------------------

    def draw_status_bar(self, imu):
        w, h = self.canvas_width, self.status_bar_height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        self.frame_count += 1
        active_states = {s.strip().lower() for s in imu.get("motion_state", [])}

        n = len(self.status_indicators)
        slot_w = w // n
        cy = h // 2

        for i, (label, color) in enumerate(self.status_indicators):
            cx = slot_w * i + 30
            active = label.lower() in active_states

            state = self.indicator_state.setdefault(label, {"active": False, "since": 0})
            if active and not state["active"]:
                # just turned on this frame -- lock the blink phase here so
                # it flashes bright immediately instead of inheriting a
                # global phase that could already be mid "off"
                state["since"] = self.frame_count
            state["active"] = active

            if active:
                elapsed = self.frame_count - state["since"]
                blink_on = (elapsed // self.blink_half_period) % 2 == 0
            else:
                blink_on = False

            if active and blink_on:
                dot_color, text_color = color, color
            elif active:
                # "off" phase of the blink -- dims out rather than disappearing,
                # so the blink reads as pulsing rather than flickering
                dot_color, text_color = (40, 40, 40), (90, 90, 90)
            else:
                dot_color, text_color = (55, 55, 55), (100, 100, 100)

            cv2.circle(canvas, (cx, cy), 9, dot_color, -1)
            cv2.circle(canvas, (cx, cy), 9, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(
                canvas, label, (cx + 18, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
            )

        return canvas

    # -------------------------------------------------
    # STITCH + DISPLAY
    # -------------------------------------------------

    def stitch(self, camera, trajectory, compass, status_bar):
        bottom = cv2.hconcat([trajectory, compass])
        return cv2.vconcat([camera, bottom, status_bar])

    def display(self, image, gnss, imu):
        camera = self.draw_camera_panel(image, gnss, imu)
        trajectory = self.draw_trajectory_panel(gnss, imu)
        compass = self.draw_compass_panel(imu)
        status_bar = self.draw_status_bar(imu)

        result = self.stitch(camera, trajectory, compass, status_bar)

        cv2.imshow(self.window_name, result)
        cv2.waitKey(1)

    def reset(self):

        self.trajectory.clear()

        self.frame_count = 0
        self.indicator_state = {}