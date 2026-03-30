# ============================================================
# save_models.py  –  Chạy 1 lần khi có mạng để tải model về
# ============================================================
import shutil
from pathlib import Path
from paddleocr import TextDetection, TextRecognition
import os

#?: Thư mục lưu model cạnh file script này (có thể đổi tuỳ ý)
LOCAL_MODEL_DIR = Path(__file__).parent / "paddle_models"

def download_and_save_models(save_dir: Path = LOCAL_MODEL_DIR):
    """
    Tải PP-OCRv5 det + rec về cache mặc định của PaddleX
    (~/.paddlex/official_models/), rồi copy sang thư mục
    save_dir để đóng gói cùng app.
    """
    #?: PaddleX cache mặc định nằm ở ~/.paddlex/official_models/
    from paddlex.utils.cache import CACHE_DIR
    paddlex_cache = Path(CACHE_DIR) / "official_models"

    models = {
        "det": "PP-OCRv5_mobile_det",
        "rec": "latin_PP-OCRv5_mobile_rec",
    }

    for role, model_name in models.items():
        dest = save_dir / model_name

        if dest.exists():
            print(f"[SKIP] {model_name} đã tồn tại tại {dest}")
            continue

        print(f"[DOWNLOAD] Đang tải {model_name} ...")

        #?: Khởi tạo model để trigger download vào PaddleX cache
        if role == "det":
            TextDetection(model_name=model_name)
        else:
            TextRecognition(model_name=model_name)

        src = paddlex_cache / model_name
        if not src.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model trong cache: {src}\n"
                "Hãy chắc chắn bạn đang có kết nối internet."
            )

        #?: Copy toàn bộ thư mục model sang save_dir
        shutil.copytree(src, dest)
        print(f"[OK] Đã lưu {model_name} → {dest}")

    print(f"\nĐã lưu xong. Cấu trúc thư mục:")
    for p in sorted(save_dir.rglob("*"))[:20]:
        print(" ", p.relative_to(save_dir))


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
    # download_and_save_models()
    det_model, rec_model = load_models()
    