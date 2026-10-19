
import sys
from pathlib import Path

THIRD_PARTY_ROOT = (
    Path.home() / "ros2_cv_ws/scripts/third_party"
)

AB3DMOT_ROOT = THIRD_PARTY_ROOT / "AB3DMOT"
XINSHUO_ROOT = THIRD_PARTY_ROOT / "Xinshuo_PyToolbox"

for path in (AB3DMOT_ROOT, XINSHUO_ROOT):
    sys.path.insert(0, str(path))

from AB3DMOT_libs.model import AB3DMOT

print("Official AB3DMOT import: PASS")
print("Tracker class:", AB3DMOT)
print("Repository:", AB3DMOT_ROOT)
