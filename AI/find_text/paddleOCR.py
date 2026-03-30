from paddleocr import TextDetection, TextRecognition
import numpy as np
import cv2

det_model = TextDetection(model_name="PP-OCRv5_mobile_det")
rec_model = TextRecognition(model_name="latin_PP-OCRv5_mobile_rec")

def crop_text_region(img, box):
    box = box.astype("float32")

    # tính width
    width_top = np.linalg.norm(box[0] - box[1])
    width_bottom = np.linalg.norm(box[2] - box[3])
    max_width = int(max(width_top, width_bottom))

    # tính height
    height_left = np.linalg.norm(box[0] - box[3])
    height_right = np.linalg.norm(box[1] - box[2])
    max_height = int(max(height_left, height_right))

    # destination box (ảnh output thẳng)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # transform
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped

img_path = "AI\\find_text\\image.png"
output = det_model.predict(input= img_path, batch_size=1)


img = cv2.imread(img_path)
output = det_model.predict(img)
for res in output:
    # res.print()
    polys = res['dt_polys'][::-1]
    for poly in polys:
        crop = crop_text_region(img, poly)
        out = rec_model.predict(crop)
        print(out[0]['rec_text'])