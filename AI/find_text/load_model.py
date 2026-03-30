# ============================================================
# load_models.py  –  Dùng trong app, chạy hoàn toàn offline
# ============================================================
import os
from pathlib import Path
from paddleocr import TextDetection, TextRecognition

#?: Trỏ đến cùng thư mục paddle_models/ đã lưu lúc download
LOCAL_MODEL_DIR = Path(__file__).parent / "paddle_models"
print(LOCAL_MODEL_DIR)
#?: Tắt hoàn toàn việc kiểm tra kết nối internet khi load
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

def load_models(model_dir: Path = LOCAL_MODEL_DIR):
    """
    Load det + rec model từ local, không cần internet.
    Trả về (det_model, rec_model).
    """
    det_path = model_dir / "PP-OCRv5_mobile_det"
    rec_path = model_dir / "latin_PP-OCRv5_mobile_rec"

    #?: Kiểm tra sớm để báo lỗi rõ ràng thay vì crash sâu bên trong
    for p in (det_path, rec_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại: {p}\n"
                "Hãy chạy save_models.py trước khi đóng gói app."
            )

    print("[LOAD] Đang load detection model ...")
    #?: Truyền model_dir= để PaddleX đọc thẳng từ local, bỏ qua download
    det_model = TextDetection(model_dir=str(det_path))

    print("[LOAD] Đang load recognition model ...")
    rec_model = TextRecognition(model_dir=str(rec_path))

    print("[OK] Load xong cả 2 model.")
    return det_model, rec_model


if __name__ == "__main__":
    det, rec = load_models()

    # Smoke test nhanh
    import numpy as np
    dummy = np.zeros((64, 320, 3), dtype=np.uint8)
    result = list(rec(dummy))
    print("Rec output sample:", result[0] if result else "(empty)")