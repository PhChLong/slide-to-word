import numpy as np
import cv2

# ============================================================
# save_models.py
# ============================================================
import os
import shutil
from pathlib import Path

#?: Phải set trước khi import paddlex để bỏ qua network check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex.inference.utils.official_models import official_models

LOCAL_MODEL_DIR = Path(__file__).parent / "paddle_models"

MODEL_NAMES = {
    "det": "PP-OCRv5_mobile_det",
    "rec": "PP-OCRv5_mobile_rec",  #?: dùng tên multilingual nếu cần tiếng Việt
}

def download_and_save_models(save_dir: Path = LOCAL_MODEL_DIR):
    """
    Chạy 1 lần khi có mạng.
    Tải model về ~/.paddlex/official_models/ rồi copy sang save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for role, model_name in MODEL_NAMES.items():
        dest = save_dir / model_name
        if dest.exists():
            print(f"[SKIP] {model_name} đã tồn tại tại {dest}")
            continue

        print(f"[DOWNLOAD] Đang tải {model_name} ...")
        #?: official_models[model_name] tự động download về _save_dir
        #?: (_save_dir = ~/.paddlex/official_models/) rồi trả về Path
        src: Path = official_models[model_name]

        if not src.exists():
            raise FileNotFoundError(
                f"Download xong nhưng không tìm thấy model tại: {src}"
            )

        shutil.copytree(src, dest)
        print(f"[OK] {model_name} → {dest}")

    print("\n[DONE] Cấu trúc thư mục:")
    for p in sorted(save_dir.rglob("*"))[:20]:
        print(" ", p.relative_to(save_dir))


# ============================================================
# ocr_models.py  –  Load offline khi chạy app
# ============================================================

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

#?: Chiến lược A: truyền model_dir trực tiếp vào create_model
#?: → PaddleX đọc thẳng từ path, KHÔNG qua _ModelManager, KHÔNG download
import paddlex

def load_models(model_dir: Path = LOCAL_MODEL_DIR):
    det_path = model_dir / MODEL_NAMES["det"]
    rec_path = model_dir / MODEL_NAMES["rec"]

    for p in (det_path, rec_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {p}\n"
                "Hãy chạy save_models.py trước khi đóng gói app."
            )

    print("[LOAD] Detection model ...")
    #?: model_dir= bypass hoàn toàn _ModelManager.official_models[]
    #?: nên sẽ không trigger download dù không có mạng
    det_model = paddlex.create_model(
        MODEL_NAMES["det"],
        model_dir=str(det_path),
    )

    print("[LOAD] Recognition model ...")
    rec_model = paddlex.create_model(
        MODEL_NAMES["rec"],
        model_dir=str(rec_path),
    )

    print("[OK] Load xong cả 2 model.")
    return det_model, rec_model

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

class PaddleOCR:
    def __init__(self, det_model, rec_model):
        self.det_model = det_model
        self.rec_model = rec_model

    def __call__(self, img):
        
        # det_model, rec_model = load_models()
        # img_path = "AI\\find_text\\image.png"

        # img = cv2.imread(img_path)
        output = self.det_model.predict(img)

        outputs = dict()
        for res in output:
            polys = res['dt_polys'][::-1]
            for i, poly in enumerate(polys):
                crop = crop_text_region(img, poly)
                # cv2.imshow("crop", crop)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                out = list(self.rec_model.predict(crop))
                text = out[0]['rec_text']
                outputs[i] = (poly, text)
        return outputs

if __name__ == "__main__":
    # download_and_save_models()
    pass