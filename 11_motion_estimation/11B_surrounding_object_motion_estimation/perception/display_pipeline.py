import cv2
import numpy as np

class DisplayPipeline:

    def __init__(self):

        self.class_names = None

    # -------------------------------------------------

    def validate(self):

        assert self.class_names is not None

    # -------------------------------------------------

    def get_motion_color(
        self,
        motion
    ):

        if motion == "Approaching":

            return (0, 0, 255)

        elif motion == "Receding":

            return (255, 0, 0)

        return (0, 255, 0)

    # -------------------------------------------------

    def draw_header(
        self,
        image,
        num_objects
    ):

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (image.shape[1], 42),
            (30, 30, 30),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            image,
            0.25,
            0,
            image
        )

        cv2.putText(
            image,
            f"Camera Perception    Objects: {num_objects}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # -------------------------------------------------
    def draw_object(
        self,
        image,
        obj
    ):

        x1, y1, x2, y2 = obj["bbox"]

        color = self.get_motion_color(
            obj["motion"]
        )

        #
        # Bounding box
        #

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        #
        # Radar association point
        #

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(
            image,
            (cx, cy),
            4,
            (255, 255, 0),
            -1
        )

        #
        # Text position
        #

        tx = x1
        ty = y1 - 72

        if ty < 20:
            ty = y2 + 20

        #
        # Object name
        #

        class_name = self.class_names[
            obj["class_id"]
        ].upper()

        cv2.putText(
            image,
            class_name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA
        )

        #
        # Divider
        #

        divider_y = ty + 8

        cv2.line(
            image,
            (tx, divider_y),
            (tx + 90, divider_y),
            color,
            2
        )

        #
        # Range
        #

        cv2.putText(
            image,
            f"{obj['range']:.1f} m",
            (tx, divider_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        #
        # Velocity
        #

        if obj["motion"] == "Approaching":

            arrow = "v"
            velocity_color = (0, 0, 255)

        elif obj["motion"] == "Receding":

            arrow = "^"
            velocity_color = (255, 0, 0)

        else:

            arrow = "o"
            velocity_color = (0, 255, 0)

        cv2.putText(
            image,
            f"{arrow} {abs(obj['velocity']):.2f}",
            (tx, divider_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            velocity_color,
            2,
            cv2.LINE_AA
        )

    def draw_legend(
        self,
        image
    ):

        x = 15
        y = image.shape[0] - 70

        cv2.putText(
            image,
            "Legend",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        y += 25

        #
        # Approaching
        #

        cv2.putText(
            image,
            "v",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0,0,255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            "Approaching",
            (x+20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

        y += 22

        #
        # Receding
        #

        cv2.putText(
            image,
            "^",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,0,0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            "Receding",
            (x+20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

        y += 22

        #
        # Stationary
        #

        cv2.putText(
            image,
            "o",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0,255,0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            "Stationary",
            (x+20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

    # -------------------------------------------------
    def draw_yolo_detections(
        self,
        image,
        boxes,
        scores,
        classes
    ):

        for box, score, cls in zip(
            boxes,
            scores,
            classes
        ):

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),   # Yellow
                1
            )

            label = self.class_names[int(cls)].upper()

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        return image

    # -------------------------------------------------

    def draw_radar_points(
        self,
        image,
        projected_points
    ):

        import time

        # blink = 3 + int(
        #     2 * np.sin(
        #         time.time() * 8.0
        #     )
        # )

        blink = 2

        for u, v, radar_target in projected_points:

            velocity = radar_target[3]

            if velocity < -0.5:

                color = (0, 0, 255)      # Red

            elif velocity > 0.5:

                color = (255, 0, 0)      # Blue

            else:

                color = (0, 255, 0)      # Green

            cv2.circle(
                image,
                (
                    int(u),
                    int(v)
                ),
                blink,
                color,
                -1
            )

            # cv2.circle(
            #     image,
            #     (
            #         int(u),
            #         int(v)
            #     ),
            #     blink + 2,
            #     color,
            #     1
            # )

        return image

    # -------------------------------------------------


    def display(
        self,
        image,
        objects
    ):

        display = image.copy()

        self.draw_header(
            display,
            len(objects)
        )

        self.draw_legend(
            display
        )

        for obj in objects:

            self.draw_object(
                display,
                obj
            )



        cv2.imshow(
            "Camera Perception",
            display
        )

        cv2.waitKey(1)