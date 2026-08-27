import sys
import numpy as np

sys.path.insert(0, "build")

import perception_cpp


pipeline = perception_cpp.Perception3DPipeline()

# --------------------------------------------------
# Synthetic mask: 1 object, 4x4
# --------------------------------------------------

masks = np.array(
    [
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    ],
    dtype=np.float32
)

# One bounding box: x1, y1, x2, y2
boxes = np.array(
    [[1, 1, 3, 3]],
    dtype=np.float32
)

# One class
classes = np.array(
    [0],
    dtype=np.int32
)

class_names = ["car"]

# Four projected LiDAR points
u = np.array([0, 1, 2, 3], dtype=np.int32)
v = np.array([0, 1, 2, 3], dtype=np.int32)

projected_points = np.array(
    [
        [10.0, 0.0, 1.5],
        [10.0, 1.0, 1.5],
        [10.0, 2.0, 1.5],
        [10.0, 3.0, 1.5],
    ],
    dtype=np.float32
)

# --------------------------------------------------
# C++ call
# --------------------------------------------------

objects = pipeline.extract_object_clouds(
    masks,
    boxes,
    classes,
    class_names,
    u,
    v,
    projected_points,
    4,
    4
)

print("Number of objects:", len(objects))

for obj in objects:

    print("Class:", obj.class_name)
    print("Box:", obj.box)

    print("Cloud:")

    for point in obj.cloud:
        print(
            "  pixel:",
            point.u,
            point.v,
            "xyz:",
            point.xyz
        )
