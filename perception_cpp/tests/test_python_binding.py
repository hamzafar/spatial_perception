import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

import perception_cpp


pipeline = perception_cpp.Perception3DPipeline()

lidar = np.array([
    [10.0,  0.0, 1.5],
    [10.0,  2.0, 1.5],
    [10.0, -2.0, 1.5],
    [-5.0,  0.0, 1.5],
], dtype=np.float32)

result = pipeline.project_lidar(
    lidar,
    "front",
    640,
    480
)

print("u:", result.u)
print("v:", result.v)
print("ego_points:")

for point in result.ego_points:
    print(point)