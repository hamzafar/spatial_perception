import cv2
import numpy as np

class DisplayPipeline:

    def __init__(self):
        self.image_width = None
        self.image_height = None

        self.bev_width = None
        self.bev_height = None
        self.scale = None  # pixels / meter

        # self.display_window_width = None
        # self.display_window_height = None

        self.class_colors = {
            "person": (0, 255, 255),
            "car": (0, 255, 0),
            "truck": (0, 0, 255),
            "bus": (255, 0, 0),
            "motorcycle": (255, 255, 0),
            "bicycle": (255, 0, 255),
        }

        self.camera_colors = {

            "front": (0, 255, 0),      # Green
            "rear": (0, 0, 255),       # Red
            "left": (255, 0, 0),       # Blue
            "right": (0, 255, 255),    # Yellow

        }

        self.unique_color = (255, 255, 255)   # White


    def initialize_display(self, display_window_width, display_window_height):

        cv2.namedWindow(
            "Phase 8 - Multi Camera Perception",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Phase 8 - Multi Camera Perception",
            display_window_width,
            display_window_height
        )
    
    def draw_camera_names(
        self,
        front,
        rear,
        left,
        right
    ):

        front = self.draw_camera_name(front, "FRONT")
        rear = self.draw_camera_name(rear, "REAR")
        left = self.draw_camera_name(left, "LEFT")
        right = self.draw_camera_name(right, "RIGHT")

        return front, rear, left, right


    def draw_bev(
        self,
        world_objects
    ):

        bev = np.zeros(
            (self.bev_height, self.bev_width, 3),
            dtype=np.uint8
        )

        center_x = bev.shape[1] // 2
        center_y = bev.shape[0] // 2

        # Ego vehicle
        cv2.rectangle(
            bev,
            (center_x - 12, center_y - 20),
            (center_x + 12, center_y + 20),
            (255, 255, 255),
            -1
        )

        # Heading arrow
        cv2.arrowedLine(
            bev,
            (center_x, center_y),
            (center_x, center_y - 35),
            (0, 255, 255),
            2,
            tipLength=0.4
        )

        # Label
        cv2.putText(
            bev,
            "EGO",
            (center_x - 15, center_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        for obj in world_objects:

            color = self.class_colors.get(
                obj["class"],
                (255, 255, 255)
            )            

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            cv2.circle(
                bev,
                (px, py),
                5,
                color,
                -1
            )

            cv2.putText(
                bev,
                obj["class"],
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, # was 0.4
                color,
                2 # was 1
            )

        return bev    

    def draw_camera_name(
        self,
        image,
        camera_name
    ):

        text_size, _ = cv2.getTextSize(
            camera_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        x = (image.shape[1] - text_size[0]) // 2
        y = 30

        cv2.putText(
            image,
            camera_name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return image

    def draw_distance(
        self,
        image,
        obj,
        distance
    ):

        if distance is None:
            return image

        x1, y1, x2, y2 = obj["box"]

        x = int((x1 + x2) / 2)
        y = int(y1)

        cv2.putText(
            image,
            f"{distance:.1f} m",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return image


    # def draw_object_cloud(
    #     self,
    #     image,
    #     object_cloud
    # ):

    #     for point in object_cloud:

    #         x, y = point["pixel"]

    #         cv2.circle(
    #             image,
    #             (x, y),
    #             2,
    #             (0, 255, 0),
    #             -1
    #         )

    #     return image

    def draw_projected_points(
        self,
        image,
        u,
        v
    ):

        for x, y in zip(u, v):

            cv2.circle(
                image,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        return image


    def draw_world_objects(
        self,
        world_objects
    ):

        bev = np.zeros(
            (self.bev_height, self.bev_width, 3),
            dtype=np.uint8
        )

        center_x = bev.shape[1] // 2
        center_y = bev.shape[0] // 2

        #
        # Ego vehicle
        #

        cv2.rectangle(
            bev,
            (center_x - 12, center_y - 20),
            (center_x + 12, center_y + 20),
            (255, 255, 255),
            -1
        )

        #
        # Heading arrow
        #

        cv2.arrowedLine(
            bev,
            (center_x, center_y),
            (center_x, center_y - 35),
            (0, 255, 255),
            2,
            tipLength=0.4
        )

        #
        # Ego label
        #

        cv2.putText(
            bev,
            "EGO",
            (center_x - 15, center_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        #
        # Draw raw camera detections
        #

        for obj in world_objects:

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            color = self.camera_colors.get(
                obj["camera"],
                (255, 255, 255)
            )

            cv2.circle(
                bev,
                (px, py),
                5,
                color,
                -1
            )

            cv2.putText(
                bev,
                obj["class"],
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        #
        # Legend
        #

        legend = [

            ("Front Cam Objects",  self.camera_colors["front"]),
            ("Rear Cam Objects",   self.camera_colors["rear"]),
            ("Left Cam Objects",   self.camera_colors["left"]),
            ("Right Cam Objects",  self.camera_colors["right"]),
            ("Merged Cam Objects", self.unique_color),

        ]

        x = 20
        y = 25

        for label, color in legend:

            cv2.circle(
                bev,
                (x, y),
                5,
                color,
                -1
            )

            cv2.putText(
                bev,
                label,
                (x + 25, y + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            y += 25

        return bev

    def draw_association_bev(
        self,
        bev,
        unique_objects
    ):

        center_x = bev.shape[1] // 2
        center_y = bev.shape[0] // 2

        for obj in unique_objects:

            if not obj.get("merged", False):
                continue

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            for src_pos in obj["source_positions"]:

                sx, sy, _ = src_pos

                spx = int(center_x + sy * self.scale)
                spy = int(center_y - sx * self.scale)

                cv2.line(
                    bev,
                    (spx, spy),
                    (px, py),
                    self.unique_color,
                    2
                )

            #
            # Draw merged object
            #

            cv2.circle(
                bev,
                (px, py),
                8,
                self.unique_color,
                -1
            )

            #
            # Draw class label
            #

            cv2.putText(
                bev,
                obj["class"],
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.unique_color,
                2
            )

        return bev


    def display_result(
        self,
        image
    ):

        cv2.imshow(
            "Phase 8 - Multi Camera Perception",
            image
        )

        cv2.waitKey(1)