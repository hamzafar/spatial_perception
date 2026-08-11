import cv2


class TrackingDisplayPipeline:

    def __init__(self):

        self.window_name = "Tracking"

        self.image_width = None
        self.image_height = None

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
        
    def display(self, image):

        cv2.imshow(
            self.window_name,
            image
        )

        cv2.waitKey(1)