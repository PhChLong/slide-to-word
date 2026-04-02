import json
import random
import shutil
from pathlib import Path

KP_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]  #?: thứ tự cố định — YOLO Pose đọc keypoints theo index, không theo tên


#@ Parse 1 task từ Label Studio JSON, trả về (raw_filename, stem, filename, yolo_line)
#@ Trả về None nếu task thiếu bbox, thiếu keypoint, hoặc chưa được annotate
def parse_task(task: dict) -> tuple | None:
    raw_filename = task["file_upload"]
    #! split("-", 1) chỉ strip đúng 1 hash prefix — nếu filename gốc có dấu "-" thì vẫn an toàn
    filename = raw_filename.split("-", 1)[1]
    stem = Path(filename).stem

    annotations = task.get("annotations", [])
    if not annotations:
        print(f"[SKIP] No annotation: {filename}")
        return None

    result = annotations[0]["result"]  #?: lấy annotation đầu tiên — thường chỉ có 1

    bbox_item = None
    kp_items = {}

    for item in result:
        if item["type"] == "rectanglelabels":
            bbox_item = item
        elif item["type"] == "keypointlabels":
            label = item["value"]["keypointlabels"][0]  #?: keypointlabels là list, luôn lấy index 0
            kp_items[label] = item

    if bbox_item is None:
        print(f"[SKIP] No bbox: {filename}")
        return None
    if len(kp_items) != 4:
        print(f"[SKIP] Expected 4 keypoints, got {len(kp_items)}: {filename}")
        return None

    bv = bbox_item["value"]
    #?: Label Studio bbox — x,y là top-left corner tính bằng % (0-100), không phải center
    bx_pct = bv["x"]
    by_pct = bv["y"]
    bw_pct = bv["width"]
    bh_pct = bv["height"]

    #?: YOLO yêu cầu cx,cy,w,h — tất cả normalize [0,1] → convert từ % sang tỉ lệ rồi tính center
    cx = (bx_pct + bw_pct / 2) / 100
    cy = (by_pct + bh_pct / 2) / 100
    bw = bw_pct / 100
    bh = bh_pct / 100

    kp_values = []
    for kp_name in KP_ORDER:
        if kp_name not in kp_items:
            print(f"[SKIP] Missing keypoint '{kp_name}': {filename}")
            return None
        kv = kp_items[kp_name]["value"]
        kp_x = kv["x"] / 100  #?: tọa độ keypoint cũng là % → normalize về [0,1]
        kp_y = kv["y"] / 100
        kp_vis = 2             #?: visibility=2: visible và labeled — chuẩn COCO/YOLO Pose
        kp_values.extend([kp_x, kp_y, kp_vis])

    #?: class index = 0 vì chỉ có 1 class (slide)
    parts = [0, cx, cy, bw, bh] + kp_values
    line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts)
    return (raw_filename, stem, filename, line)


#@ Convert toàn bộ Label Studio JSON export → dataset YOLO Pose format
#@ Tạo cấu trúc dataset/train/ và dataset/val/ với images/ và labels/ bên trong
#@ Tự động shuffle và split theo val_ratio, copy ảnh gốc vào đúng thư mục
def convert(
    json_path: str,
    images_dir: str,        #?: thư mục upload của Label Studio — thường ở %LOCALAPPDATA%/label-studio/media/upload/1/
    output_dir: str,
    val_ratio: float = 0.2  #?: tỉ lệ ảnh dùng làm validation
):
    json_path  = Path(json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    converted = [r for task in tasks if (r := parse_task(task)) is not None]  #?: walrus operator — parse và filter None trong 1 bước

    random.seed(42)  #?: seed cố định để split train/val reproducible
    random.shuffle(converted)
    n_val     = max(1, int(len(converted) * val_ratio))
    val_set   = converted[:n_val]
    train_set = converted[n_val:]

    for split, items in [("train", train_set), ("val", val_set)]:
        for raw_filename, stem, filename, line in items:
            src_img = images_dir / filename
            dst_img = output_dir / split / "images" / filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                print(f"[WARN] Image not found: {src_img}")

            label_path = output_dir / split / "labels" / f"{stem}.txt"
            with open(label_path, "w") as f:
                f.write(line + "\n")

    print(f"\nDone: {len(train_set)} train, {len(val_set)} val")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    BASE = Path(__file__).parent  #?: __file__ là path của script này → .parent là thư mục chứa nó
    convert(
        json_path  = BASE / "annotations.json",
        images_dir = "D:/Github Repos/slide-to-word/data/",
        output_dir = BASE / "dataset/",
    )
    #@ convert theo format:
    """
    <class> <cx> <cy> <bw> <bh> <kp1_x> <kp1_y> <kp1_vis> <kp2_x> <kp2_y> <kp2_vis> <kp3_x> <kp3_y> <kp3_vis> <kp4_x> <kp4_y> <kp4_vis>
    ex: 0  cx  cy  bw  bh  top_left_x  top_left_y  2  top_right_x  top_right_y  2  bottom_right_x  bottom_right_y  2  bottom_left_x  bottom_left_y  2
    kp_vis = 2 là visibility flag — một field bắt buộc trong format YOLO Pose, mượn từ chuẩn COCO keypoints.
    3 giá trị có thể:

    0 — keypoint không tồn tại / không annotate
    1 — keypoint bị che khuất (occluded), tọa độ vẫn được estimate
    2 — keypoint visible và được label chính xác
    """
    