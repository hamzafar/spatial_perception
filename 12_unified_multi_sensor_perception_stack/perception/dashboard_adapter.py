import base64
import cv2
import numpy as np

class DashboardAdapter:

    def __init__(self, class_names):
        self.class_names = class_names

    # -------------------------------------------------

    def prepare_tracked_camera(
        self,
        image,
        online_targets,
        boxes,
        scores,
        classes,
        image_width,
        image_height,
        camera_prefix
    ):
        tracked_boxes = []
        if len(boxes) == 0 or len(online_targets) == 0:
            return self.prepare_image(image, tracked_boxes)

        boxes = np.asarray(boxes, dtype=np.float64)          # (N, 4) x1,y1,x2,y2
        det_x1, det_y1, det_x2, det_y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        det_areas = (det_x2 - det_x1) * (det_y2 - det_y1)     # (N,)

        for target in online_targets:

            tx1, ty1, tw, th = map(float, target.tlwh)
            tx2, ty2 = tx1 + tw, ty1 + th
            track_area = tw * th

            # vectorized IoU against every detection at once
            ix1 = np.maximum(tx1, det_x1)
            iy1 = np.maximum(ty1, det_y1)
            ix2 = np.minimum(tx2, det_x2)
            iy2 = np.minimum(ty2, det_y2)
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

            union = track_area + det_areas - inter
            iou = np.where(union > 0, inter / union, 0.0)

            best_idx = int(np.argmax(iou))
            if iou[best_idx] <= 0.0:
                continue

            cls = int(classes[best_idx])
            score = float(scores[best_idx])

            tracked_boxes.append({
                "id": f"{camera_prefix}_{int(target.track_id)}",
                "cls": self.class_names[cls],
                "conf": score,
                "box": [
                    ((tx1 + tx2) / 2) / image_width,
                    ((ty1 + ty2) / 2) / image_height,
                    (tx2 - tx1) / image_width,
                    (ty2 - ty1) / image_height,
                ],
            })

        return self.prepare_image(image, tracked_boxes)

    def prepare_tracked_camera_2fl(
        self,
        image,
        online_targets,
        boxes,
        scores,
        classes,
        image_width,
        image_height
    ):

        tracked_boxes = []

        for target in online_targets:

            tlwh = target.tlwh

            tx1 = float(tlwh[0])
            ty1 = float(tlwh[1])
            tw = float(tlwh[2])
            th = float(tlwh[3])

            tx2 = tx1 + tw
            ty2 = ty1 + th

            track_id = int(target.track_id)

            best_iou = 0.0
            best_idx = -1

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(float, box)

                ix1 = max(tx1, x1)
                iy1 = max(ty1, y1)
                ix2 = min(tx2, x2)
                iy2 = min(ty2, y2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)

                intersection = iw * ih

                track_area = tw * th
                detection_area = (
                    (x2 - x1) *
                    (y2 - y1)
                )

                union = (
                    track_area +
                    detection_area -
                    intersection
                )

                iou = (
                    intersection / union
                    if union > 0
                    else 0.0
                )

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx < 0:
                continue

            cls = int(classes[best_idx])
            score = float(scores[best_idx])

            cx = (
                (tx1 + tx2) / 2
            ) / image_width

            cy = (
                (ty1 + ty2) / 2
            ) / image_height

            bw = (
                tx2 - tx1
            ) / image_width

            bh = (
                ty2 - ty1
            ) / image_height

            tracked_boxes.append({
                "id": track_id,
                "cls": self.class_names[cls],
                "conf": score,
                "box": [cx, cy, bw, bh]
            })

        return self.prepare_image(
            image,
            tracked_boxes
        )

    # -------------------------------------------------

    def prepare_image(
        self,
        image,
        boxes
    ):
        success, buffer = cv2.imencode(
            ".jpg",
            image
        )

        if not success:
            raise RuntimeError(
                "Failed to encode camera image"
            )

        image_b64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        return {
            "image": image_b64,
            "boxes": boxes
        }

    # -------------------------------------------------

    def prepare_camera(
        self,
        image,
        boxes,
        scores,
        classes
    ):

        h, w = image.shape[:2]

        dashboard_boxes = []

        for box, score, cls in zip(
            boxes,
            scores,
            classes
        ):

            x1, y1, x2, y2 = box

            cx = (
                (x1 + x2) / 2
            ) / w

            cy = (
                (y1 + y2) / 2
            ) / h

            bw = (
                x2 - x1
            ) / w

            bh = (
                y2 - y1
            ) / h

            class_id = int(cls)

            class_name = self.class_names[class_id]

            dashboard_boxes.append({
                "cls": class_name,
                "id": 0,
                "conf": float(score),
                "box": [
                    cx,
                    cy,
                    bw,
                    bh
                ]
            })

        return self.prepare_image(
            image,
            dashboard_boxes
        )