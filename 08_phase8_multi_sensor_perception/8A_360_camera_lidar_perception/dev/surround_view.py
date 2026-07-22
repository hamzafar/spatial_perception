import cv2
import numpy as np


class SurroundView:

    def __init__(self,
                 canvas_size=1200,
                 ego_radius=250):

        self.size = canvas_size
        self.ego_radius = ego_radius

        self.src = np.float32([
            [0, 0],
            [639, 0],
            [639, 479],
            [0, 479]
        ])

        c = canvas_size // 2
        r = ego_radius

        self.dst = {

            "front": np.float32([
                [0, 0],
                [canvas_size, 0],
                [c + r, c + 20],
                [c - r, c + 20]
            ]),

            "rear": np.float32([
                [c - r, c + r],
                [c + r, c + r],
                [canvas_size, canvas_size],
                [0, canvas_size]
            ]),

            # "left": np.float32([
            #     [0, 0],
            #     [c - r + 20, c - r],
            #     [c - r + 20, c + r],
            #     [0, canvas_size]
            # ]),
            "left": np.float32([
                [0, 0],
                [c-r+50, c-r -20],
                [c-r+50, c+r+20],
                [0, canvas_size]
            ]),

            "right": np.float32([
                [c + r - 20, c - r],
                [canvas_size, 0],
                [canvas_size, canvas_size],
                [c + r - 20, c + r]
            ])
        }

    ###########################################################################

    def compose(self, front, rear, left, right):

        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        warps = [
            self.warp_camera_section(front, "front"),
            self.warp_camera_section(rear, "rear"),
            self.warp_camera_section(left, "left"),
            self.warp_camera_section(right, "right")
        ]

        for warped in warps:
            mask = np.any(warped != 0, axis=2)
            canvas[mask] = warped[mask]

        return canvas

    # def compose(self, front, rear, left, right):

    #     canvas = np.zeros(
    #         (self.size, self.size, 3),
    #         dtype=np.uint8
    #     )

    #     canvas = self.warp_camera(canvas, front, "front")
    #     canvas = self.warp_camera(canvas, rear, "rear")
    #     canvas = self.warp_camera(canvas, left, "left")
    #     canvas = self.warp_camera(canvas, right, "right")

    #     self.draw_separators(canvas)
    #     # self.draw_ego(canvas)
    #     self.draw_labels(canvas)

    #     return canvas

    # ###########################################################################

    def warp_camera_section(self, image, camera):

        H = cv2.getPerspectiveTransform(
            self.src,
            self.dst[camera]
        )

        return cv2.warpPerspective(
            image,
            H,
            (self.size, self.size)
        )
    # def warp_camera(self, canvas, image, camera):

    #     H = cv2.getPerspectiveTransform(self.src, self.dst[camera])

    #     warped = cv2.warpPerspective(
    #         image,
    #         H,
    #         (self.size,self.size)
    #     )

    #     cv2.imshow(camera, warped)
    #     cv2.waitKey(1)

    #     return canvas

    # def warp_camera_section(self, canvas, image, camera):

    #     H = cv2.getPerspectiveTransform(self.src, self.dst[camera])

    #     warped = cv2.warpPerspective(
    #         image,
    #         H,
    #         (self.size, self.size)
    #     )

    #     return warped

    # def warp_camera(self, canvas, image, camera):

    #     h = 300
    #     w = 400

    #     img = cv2.resize(image, (w, h))

    #     if camera == "front":
    #         canvas[0:h, 400:800] = self.warp_camera_section(canvas[0:h, 400:800],img, camera)

    #     # elif camera == "rear":
    #     #     canvas[900:1200, 400:800] = self.warp_camera_section(canvas,img, camera)

    #     # elif camera == "left":
    #     #     canvas[450:750, 0:400] = img

    #     # elif camera == "right":
    #     #     canvas[450:750, 800:1200] = img

    #     return canvas
    ###########################################################################

    def draw_ego(self,
                 canvas):

        c = self.size // 2

        cv2.circle(
            canvas,
            (c, c),
            self.ego_radius,
            (35, 35, 35),
            -1
        )

        cv2.circle(
            canvas,
            (c, c),
            self.ego_radius,
            (255, 255, 255),
            2
        )

        cv2.arrowedLine(
            canvas,
            (c, c + 25),
            (c, c - 45),
            (0, 255, 255),
            3,
            tipLength=0.35
        )

    ###########################################################################

    def draw_labels(self,
                    canvas):

        c = self.size // 2

        cv2.putText(
            canvas,
            "FRONT",
            (c - 55, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            "REAR",
            (c - 40, self.size - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            "LEFT",
            (40, c),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            "RIGHT",
            (self.size - 150, c),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    ###########################################################################

    def draw_separators(self,
                        canvas):

        c = self.size // 2

        cv2.line(
            canvas,
            (0, 0),
            (c, c),
            (255, 255, 255),
            2
        )

        cv2.line(
            canvas,
            (self.size, 0),
            (c, c),
            (255, 255, 255),
            2
        )

        cv2.line(
            canvas,
            (0, self.size),
            (c, c),
            (255, 255, 255),
            2
        )

        cv2.line(
            canvas,
            (self.size, self.size),
            (c, c),
            (255, 255, 255),
            2
        )