from pathlib import Path
import cv2

from AI.find_text import PaddleOCR, load_models

det_model, rec_model = load_models()

ocr = PaddleOCR(det_model, rec_model)

data_dir = Path("data")
outputs = []
# image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# for image_path in data_dir.iterdir():
#     if image_path.is_file() and image_path.suffix.lower() in image_extensions:
#         img = cv2.imread(image_path)
#         #@ yolo detect_crop ảnh
#         #! trống

#         #@ trích xuất text
#         out = ocr(img)
#         outputs.append(out) #@ out = dict(index: (bbox, text))
image_path = "image.png"
img = cv2.imread(image_path)
out = ocr(img)
print(out)
