import cv2
import numpy as np

class Tracking3DDisplayPipeline:

    def __init__(self):

        self.window_name = "Tracking"

        self.image_width = None
        self.image_height = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def initialize_display(
        self,
        display_window_width,
        display_window_height
    ):

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            self.window_name,
            display_window_width,
            display_window_height
        )

    def draw_tracks(
        self,
        image,
        online_targets,
        box_color=(0, 255, 0),
        thickness=2,
    ):

        for track in online_targets:

            x1, y1, x2, y2 = map(int, track.tlbr)

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness
            )

            # Keep label inside the image
            label_y = max(20, y1 - 10)

            cv2.putText(
                image,
                f"ID {track.track_id}",
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA
            )

        return image

    def compute_box_corners(
        self,
        center,
        dimensions,
        yaw
    ):
        """
        Generate 8 corners of a 3D bounding box.

        Args:
            center: (x, y, z)
            dimensions: (length, width, height)
            yaw: radians

        Returns:
            (8,3) numpy array
        """

        length, width, height = dimensions

        l = length / 2.0
        w = width  / 2.0
        h = height / 2.0

        #
        # Local corners
        #

        corners = np.array([
            [ l,  w,  h],
            [ l, -w,  h],
            [-l, -w,  h],
            [-l,  w,  h],
            [ l,  w, -h],
            [ l, -w, -h],
            [-l, -w, -h],
            [-l,  w, -h]
        ])

        #
        # Rotation around vertical axis
        #

        c = np.cos(yaw)
        s = np.sin(yaw)

        R = np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1]
        ])

        corners = corners @ R.T

        corners += center

        return corners

    def project_box(
        self,
        corners
    ):
        """
        Project 3D corners into image.
        """

        projected = []

        for i, corner in enumerate(corners):
            
            X, Y, Z = corner
            # print(f"\nCorner {i}")
            # print("X =", X)
            # print("Y =", Y)
            # print("Z =", Z)

            #
            # Behind camera
            #

            # if Z <= 0.1:
            #     return None

            u = self.fx * X / Z + self.cx
            v = self.fy * Y / Z + self.cy

            projected.append(
                (
                    int(u),
                    int(v)
                )
            )

        return projected

    def draw_box(
        self,
        image,
        pts,
        color=(0,255,0),
        thickness=2
    ):
        """
        Draw projected cuboid.
        """

        if pts is None:
            return

        edges = [

            #
            # Top
            #

            (0,1),
            (1,2),
            (2,3),
            (3,0),

            #
            # Bottom
            #

            (4,5),
            (5,6),
            (6,7),
            (7,4),

            #
            # Vertical
            #

            (0,4),
            (1,5),
            (2,6),
            (3,7)
        ]

        for i, j in edges:

            cv2.line(
                image,
                pts[i],
                pts[j],
                color,
                thickness
            )

    def draw_3d_boxes(
        self,
        image,
        objects
    ):
        """
        Draw 3D detections or tracked objects.

        Expected object format:
        {
            "center": np.ndarray(3,),
            "dimensions": (length, width, height),
            "yaw": float,

            # Optional
            "track_id": int,
            "class": str
        }
        """

        for obj in objects:

            corners = self.compute_box_corners(
                obj["center"],
                obj["dimensions"],
                obj["yaw"]
            )

            corners = self.transform_to_optical_frame(
                corners
            )

            projected = self.project_box(
                corners
            )

            if projected is None:
                continue

            #
            # Draw 3D cuboid
            #

            self.draw_box(
                image,
                projected
            )

            #
            # Draw track ID (only for tracked objects)
            #

            if "track_id" in obj:

                x = min(pt[0] for pt in projected)
                y = min(pt[1] for pt in projected)

                if "class" in obj:
                    label = f'{obj["class"]} #{obj["track_id"]}'
                else:
                    label = f'ID {obj["track_id"]}'

                cv2.putText(
                    image,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


    def draw_3d_boxes_only(
        self,
        image,
        detections
    ):
        """
        Draw all 3D detections with class names.
        """

        for det in detections:

            corners = self.compute_box_corners(
                det["center"],
                det["dimensions"],
                det["yaw"]
            )

            corners = self.transform_to_optical_frame(
                corners
            )

            projected = self.project_box(
                corners
            )

            if projected is None:
                continue

            self.draw_box(
                image,
                projected
            )

            # Draw class name
            x = max(0, min(p[0] for p in projected))
            y = max(20, min(p[1] for p in projected))

            # print(det["class"], x, y)

            cv2.putText(
                image,
                det["class"],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            
    def transform_to_optical_frame(self, corners):
        """
        Convert 3D points from perception frame to
        OpenCV camera optical frame.

        Perception Frame:
            X = Forward
            Y = Left / Right
            Z = Up

        Camera Optical Frame:
            X = Right
            Y = Down
            Z = Forward

        Parameters
        ----------
        corners : np.ndarray
            Shape (N,3)

        Returns
        -------
        np.ndarray
            Shape (N,3) in camera optical frame.
        """

        corners = np.asarray(corners)

        optical = np.empty_like(corners)

        optical[:, 0] = corners[:, 1]      # X_cam = Y_world
        optical[:, 1] = -corners[:, 2]     # Y_cam = -Z_world
        optical[:, 2] = corners[:, 0]      # Z_cam = X_world

        return optical

    def draw_3d_objects_ids(
        self,
        image,
        objects,
        color=(0, 255, 0)
    ):

        for obj in objects:

            corners = self.compute_box_corners(
                obj["center"],
                obj["dimensions"],
                obj["yaw"]
            )

            corners = self.transform_to_optical_frame(
                corners
            )

            projected = self.project_box(
                corners
            )

            if projected is None:
                continue

            self.draw_box(
                image,
                projected,
                color=color
            )

            if "track_id" in obj:

                x = min(pt[0] for pt in projected)
                y = min(pt[1] for pt in projected)

                cv2.putText(
                    image,
                    f'ID {obj["track_id"]}',
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )


    def draw_3d_centers(
        self,
        image,
        detections,
        color=(0, 0, 255)
    ):
        """
        Draw only the projected 3D center of each detection.
        """

        for det in detections:

            center = np.asarray(det["center"]).reshape(1, 3)

            center = self.transform_to_optical_frame(
                center
            )

            projected = self.project_box(
                center
            )

            if projected is None:
                continue

            x, y = projected[0]

            cv2.circle(
                image,
                (x, y),
                5,
                color,
                -1
            )

            # Get class name
            class_name = det.get("class", "Unknown")

            # Draw class name
            cv2.putText(
                image,
                class_name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )

    def draw_yolo_detections(
        self,
        image,
        boxes,
        scores=None,
        classes=None,
        color=(255, 0, 0)
    ):
        """
        Draw YOLO bounding boxes and their centers.

        Args:
            image: OpenCV image.
            boxes: Nx4 array of [x1, y1, x2, y2].
            scores: Optional confidence scores.
            classes: Optional class ids.
            color: BGR color (default blue).
        """

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box)

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Box center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                image,
                (cx, cy),
                5,
                color,
                -1
            )

            # Optional confidence
            if scores is not None:
                cv2.putText(
                    image,
                    f"{scores[i]:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return image

    def display(self, image):

        cv2.imshow(
            self.window_name,
            image
        )

        cv2.waitKey(1)