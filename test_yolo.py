import torch
from ultralytics import YOLO

model = YOLO("runs/pose/train4/weights/best.pt")
import cv2
import numpy as np

path = "received_1275287874043041.jpeg"
img = cv2.imread(path)
h, w = img.shape[:2]

keypoints = model("received_1275287874043041.jpeg")[0].keypoints.xyn.cpu().numpy().squeeze()

pixel_points = (keypoints * np.array([w, h])).astype(np.int32)
cv2.polylines(img, [pixel_points.reshape(-1, 1, 2)], isClosed= True, color = (0, 255, 0), thickness= 2)
for x, y in pixel_points:
    cv2.circle(img, (x, y), radius= 5, color = (0, 0, 255), thickness= -1)
img_resized = cv2.resize(img, (800, 600))
cv2.imshow("result", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(pixel_points)