from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO("runs/pose/train4/weights/best.pt")

@app.post("/detect-corners-batch")
async def detect_corners(images: list[UploadFile] = [File(...)]):
    # Đọc raw bytes từ request
    data = list()
    for i, image in enumerate(images):
        raw_byte = await image.read()
    
        # Decode bytes → numpy array (giống cv2.imread nhưng từ memory)
        np_arr = np.frombuffer(raw_byte, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  #? imdecode thay cho imread vì input là bytes, không phải path
    
        results = model(img)[0]
        keypoints = results.keypoints.xyn.cpu().numpy().squeeze()
    
        # Trả về corners dạng dict để JS dùng
        tl, tr, br, bl = keypoints  
        data.append(
            {
                "tl": {"x": float(tl[0]), "y": float(tl[1])},
                "tr": {"x": float(tr[0]), "y": float(tr[1])},
                "br": {"x": float(br[0]), "y": float(br[1])},
                "bl": {"x": float(bl[0]), "y": float(bl[1])},
            }
        )
    return data
        