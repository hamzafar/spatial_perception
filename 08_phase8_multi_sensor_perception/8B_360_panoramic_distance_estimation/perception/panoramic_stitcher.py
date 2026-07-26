import cv2
import numpy as np


class PanoramicStitcher:
    """
    Projects four perspective camera images (front/rear/left/right) onto a
    shared cylindrical panorama, using each camera's known yaw and pinhole
    intrinsics (fx, fy, cx, cy).

    The per-pixel source lookup (which camera pixel a given panorama pixel
    comes from) only depends on fixed camera geometry, so it is computed
    once in __init__ and reused every frame via cv2.remap. Overlapping
    regions between adjacent cameras are alpha-feathered to avoid hard
    seams.
    """

    def __init__(
        self,
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
        camera_yaws,
        pano_width=1920,
        pano_height=600,
        vertical_fov_deg=80.0,
        blend_overlap_deg=6.0
    ):

        self.image_width = image_width
        self.image_height = image_height

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # e.g. {"front": 0, "rear": 180, "left": -90, "right": 90}
        self.camera_yaws = camera_yaws

        self.pano_width = pano_width
        self.pano_height = pano_height

        self.vertical_fov = np.deg2rad(vertical_fov_deg)
        self.blend_overlap = np.deg2rad(blend_overlap_deg)

        # Camera horizontal half-FOV, derived from intrinsics.
        self.half_fov_h = np.arctan2(
            self.image_width / 2.0,
            self.fx
        )

        self.remap_tables = {}
        self.blend_weights = {}

        self._build_remap_tables()
        self.vignette_mask = self._build_vignette_mask()

    # -------------------------------------------------
    # Precompute, once, which source pixel each panorama
    # pixel should sample from, per camera.
    # -------------------------------------------------

    def _build_remap_tables(self):

        # Panorama columns span 0..2*pi azimuth, centered on "forward".
        cols = np.arange(self.pano_width)
        azimuth = (cols / self.pano_width) * 2.0 * np.pi - np.pi

        rows = np.arange(self.pano_height)
        elevation = (0.5 - rows / self.pano_height) * self.vertical_fov

        azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)

        for camera_name, yaw_deg in self.camera_yaws.items():

            yaw = np.deg2rad(yaw_deg)

            # Angle of this panorama direction relative to the camera's
            # own optical axis. Note the minus sign: with this rig's
            # rotation convention (same R(yaw) used in project_lidar),
            # a camera's forward direction in world azimuth is -yaw, not
            # +yaw. Using "+yaw" here silently swaps left/right.
            beta = azimuth_grid - yaw
            beta = np.arctan2(np.sin(beta), np.cos(beta))  # wrap to [-pi, pi]

            cos_beta = np.cos(beta)

            valid = cos_beta > 1e-4  # point must be in front of the camera

            u = np.full_like(beta, -1.0, dtype=np.float32)
            v = np.full_like(beta, -1.0, dtype=np.float32)

            tan_beta = np.tan(beta)
            tan_elev = np.tan(elevation_grid)

            u[valid] = (
                self.fx * tan_beta[valid] + self.cx
            ).astype(np.float32)

            v[valid] = (
                self.fy * (-tan_elev[valid]) / cos_beta[valid] + self.cy
            ).astype(np.float32)

            in_bounds = (
                valid &
                (u >= 0) & (u < self.image_width) &
                (v >= 0) & (v < self.image_height)
            )

            u[~in_bounds] = -1.0
            v[~in_bounds] = -1.0

            self.remap_tables[camera_name] = (u, v)

            # Feather weight: fades to 0 near this camera's FOV edge so
            # overlapping cameras blend instead of producing a hard seam.
            edge_distance = self.half_fov_h - np.abs(beta)

            weight = np.clip(
                edge_distance / max(self.blend_overlap, 1e-6),
                0.0,
                1.0
            )

            weight[~in_bounds] = 0.0

            self.blend_weights[camera_name] = weight.astype(np.float32)

    # -------------------------------------------------

    def _build_vignette_mask(self, strength=0.25):
        """
        Soft per-column darkening near each camera's own edge, so any
        residual brightness/exposure mismatch between adjacent cameras
        is disguised rather than showing as a visible seam.
        """

        combined = np.zeros(
            (self.pano_height, self.pano_width),
            dtype=np.float32
        )

        for weight in self.blend_weights.values():
            combined = np.maximum(combined, weight)

        # combined is 1.0 in the confident center of a camera's FOV and
        # fades to 0 near its edge -- invert that into a mild darkening
        # factor so edges dim slightly instead of just blending flatly.
        darken = 1.0 - strength * (1.0 - combined)

        return darken.astype(np.float32)

    # -------------------------------------------------

    def convert_panoramic_images(self, front, rear, left, right):

        images = {
            "front": front,
            "rear": rear,
            "left": left,
            "right": right
        }

        canvas = np.zeros(
            (self.pano_height, self.pano_width, 3),
            dtype=np.float32
        )

        weight_sum = np.zeros(
            (self.pano_height, self.pano_width),
            dtype=np.float32
        )

        for camera_name, image in images.items():

            u, v = self.remap_tables[camera_name]
            weight = self.blend_weights[camera_name]

            warped = cv2.remap(
                image,
                u,
                v,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            canvas += warped.astype(np.float32) * weight[..., None]
            weight_sum += weight

        # Avoid divide-by-zero in any uncovered gap between cameras.
        weight_sum[weight_sum == 0] = 1.0

        panorama = canvas / weight_sum[..., None]
        panorama = panorama * self.vignette_mask[..., None]
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        return panorama

    # -------------------------------------------------

    def azimuth_to_column(self, azimuth_rad):
        """
        Convert a world-frame azimuth angle (radians, 0 = straight ahead,
        matching the LiDAR frame's atan2(y, x)) into a panorama column
        index. Use this to anchor per-object distance labels on the
        panorama instead of on each source camera image.
        """

        azimuth_rad = np.arctan2(
            np.sin(azimuth_rad),
            np.cos(azimuth_rad)
        )

        col = ((azimuth_rad + np.pi) / (2.0 * np.pi)) * self.pano_width

        return int(np.clip(col, 0, self.pano_width - 1))

    # -------------------------------------------------
    # HUD overlays -- all drawn AFTER convert_panoramic_images,
    # directly on the finished panorama. Never draw text on a
    # per-camera image before warping: cv2.remap stretches text
    # exactly like scene content, so labels near a camera's edge
    # come out curved/smeared.
    # -------------------------------------------------

    def draw_compass_strip(self, panorama, camera_yaws=None, tick_step_deg=30):
        """
        Thin heading strip along the top edge: tick marks every
        tick_step_deg, plus each camera's name positioned by its actual
        yaw (via azimuth_to_column) instead of a hard-coded x/y per
        camera image.
        """

        strip_height = 22

        overlay = panorama.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (self.pano_width, strip_height),
            (30, 30, 30),
            -1
        )

        cv2.addWeighted(overlay, 0.55, panorama, 0.45, 0, panorama)

        for deg in range(-180, 180, tick_step_deg):

            col = self.azimuth_to_column(np.deg2rad(deg))

            cv2.line(
                panorama,
                (col, 0),
                (col, 6),
                (150, 150, 150),
                1,
                cv2.LINE_AA
            )

        if camera_yaws:

            for name, yaw_deg in camera_yaws.items():

                col = self.azimuth_to_column(np.deg2rad(yaw_deg))

                label = name.upper()

                text_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )

                x = int(np.clip(
                    col - text_size[0] // 2,
                    2,
                    self.pano_width - text_size[0] - 2
                ))

                cv2.putText(
                    panorama,
                    label,
                    (x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (249, 200, 120),
                    1,
                    cv2.LINE_AA
                )

        return panorama

    # -------------------------------------------------

    def _distance_color(self, distance, near=10.0, far=25.0):
        """
        Smooth red -> yellow -> green gradient over [near, far] meters,
        instead of three hard-cutoff buckets. Returns a BGR tuple.
        """

        t = np.clip((distance - near) / max(far - near, 1e-6), 0.0, 1.0)

        if t < 0.5:
            # red -> yellow
            local_t = t / 0.5
            b, g, r = 0, int(255 * local_t), 255
        else:
            # yellow -> green
            local_t = (t - 0.5) / 0.5
            b, g, r = 0, 255, int(255 * (1.0 - local_t))

        return (b, g, r)

    # -------------------------------------------------

    def draw_distance_pill(self, panorama, azimuth_rad, distance, label_offset_y=-14):
        """
        Draws a filled rounded-rectangle "pill" badge with the distance
        text on top, anchored to an object's real-world azimuth on the
        panorama -- readable over both bright and dark backgrounds.
        """

        if distance is None:
            return panorama

        col = self.azimuth_to_column(azimuth_rad)
        row = self.pano_height // 2 + label_offset_y

        text = f"{distance:.1f} m"
        color = self._distance_color(distance)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        pad_x, pad_y = 8, 5

        x0 = col - w // 2 - pad_x
        y0 = row - h - pad_y
        x1 = col + w // 2 + pad_x
        y1 = row + pad_y

        overlay = panorama.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            color,
            -1,
            cv2.LINE_AA
        )

        cv2.addWeighted(overlay, 0.55, panorama, 0.45, 0, panorama)

        cv2.putText(
            panorama,
            text,
            (col - w // 2, row - pad_y // 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        return panorama

    # -------------------------------------------------

    def draw_ego_marker(self, panorama):
        """
        Small semi-transparent marker at the bottom-center, so distance
        readouts have a visible reference point ("measured from here")
        instead of just floating text.
        """

        col = self.pano_width // 2
        row = self.pano_height - 6

        overlay = panorama.copy()

        pts = np.array(
            [
                [col - 26, row],
                [col + 26, row],
                [col + 16, row - 14],
                [col - 16, row - 14]
            ],
            dtype=np.int32
        )

        cv2.fillPoly(overlay, [pts], (60, 200, 255))

        cv2.addWeighted(overlay, 0.5, panorama, 0.5, 0, panorama)

        return panorama

    # -------------------------------------------------

    def draw_distance_legend(self, panorama):
        """
        Fixed HUD legend in the top-left corner of the finished
        panorama -- always visible, independent of any single camera.
        """

        legend = [
            ("Safe", (0, 255, 0)),
            ("Caution", (0, 255, 255)),
            ("Danger", (0, 0, 255))
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        x0, y0 = 8, 28
        row_height = 18

        box_height = row_height * len(legend) + 10
        box_width = 90

        overlay = panorama.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (30, 30, 30),
            -1
        )

        cv2.addWeighted(overlay, 0.55, panorama, 0.45, 0, panorama)

        y = y0 + 16

        for label, color in legend:

            cv2.circle(panorama, (x0 + 12, y - 4), 5, color, -1, cv2.LINE_AA)

            cv2.putText(
                panorama,
                label,
                (x0 + 24, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            y += row_height

        return panorama