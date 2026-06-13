import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.face_test import detect_face_bbox

img_path = Path(__file__).parent.parent / "data" / "bbox_test.jpg"
img = cv2.imread(str(img_path))

bbox = detect_face_bbox(img)

if bbox is None:
    print("未检测到人脸")
else:
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"检测到人脸: x={x}, y={y}, w={w}, h={h}")

cv2.imshow("bbox test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()