import json
from pathlib import Path

#@ Convert COCO keypoint JSON (Roboflow export) sang YOLOv8 Pose .txt format
#@ Keypoint order: top_left, top_right, bottom_right, bottom_left (clockwise)
#@ Output: mỗi ảnh 1 file .txt cùng tên, trong thư mục labels/

def convert(json_path: str, output_dir: str):
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    #? Build lookup: image_id -> (width, height, file_name)
    image_info = {
        img["id"]: (img["width"], img["height"], img["file_name"])
        for img in data["images"]
    }

    #? Build lookup: image_id -> list of annotations
    annots_by_image = {}
    for ann in data["annotations"]:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    skipped = []
    converted = 0

    for image_id, annots in annots_by_image.items():
        w, h, file_name = image_info[image_id]
        
        lines = []
        for ann in annots:
            kps = ann.get("keypoints", [])
            
            #! Skip nếu không đủ 4 keypoints (12 values)
            if len(kps) < 12:
                skipped.append((file_name, ann["id"], len(kps) // 3))
                continue

            #? Parse 4 keypoints: [x1,y1,v1, x2,y2,v2, x3,y3,v3, x4,y4,v4]
            points = [(kps[i], kps[i+1], kps[i+2]) for i in range(0, 12, 3)]

            #? Tính bbox từ keypoints (min/max của 4 góc)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            #? Normalize bbox về [0,1] theo YOLOv8 format: cx cy bw bh
            cx = ((x_min + x_max) / 2) / w
            cy = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            #? Normalize keypoints về [0,1], giữ visibility
            kp_str = " ".join(
                f"{x/w:.6f} {y/h:.6f} {int(v)}"
                for x, y, v in points
            )

            #? class_id = 0 (chỉ có 1 class: slide)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}")

        #? Tên file .txt = tên ảnh bỏ extension
        stem = Path(file_name).stem
        out_file = output_dir / f"{stem}.txt"
        
        if lines:
            out_file.write_text("\n".join(lines))
            converted += 1
        else:
            #? Tạo file rỗng để YOLO không báo missing label
            out_file.write_text("")

    print(f"Done: {converted} files written to {output_dir}")
    if skipped:
        print(f"\nSkipped {len(skipped)} annotations (keypoints < 4):")
        for fname, ann_id, n_kp in skipped:
            print(f"  ann_id={ann_id} | {fname} | got {n_kp} keypoints")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_coco_to_yolo_pose.py <coco.json> <output_labels_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])