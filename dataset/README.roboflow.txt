detect_slide_corners - vdataset pham-chau-long

The dataset includes 218 images.
Detect_slide_corners are annotated in COCO format.

No pre-processing or augmentation was applied.

# YOLOv8 Pose — Slide Corner Detection

Detect 4 góc của slide trong ảnh bằng YOLOv8 Pose.  
Mỗi slide = 1 object với 4 keypoints theo thứ tự clockwise:  
`top_left → top_right → bottom_right → bottom_left`

---

## Cấu trúc project

```
project/
├── roboflow/
│   ├── train/          # ảnh gốc (.jpg / .png)
│   └── label/          # label files (.txt) — 1 file per ảnh
├── convert_coco_to_yolo_pose.py   # convert Roboflow COCO JSON → YOLO .txt
├── train_pose.py                  # train script
└── README.md
```

---

## Format label (.txt)

Mỗi file `.txt` tương ứng 1 ảnh. Mỗi dòng = 1 object (1 slide).

```
<class_id> <cx> <cy> <bw> <bh> <kp0_x> <kp0_y> <kp0_v> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> <kp3_x> <kp3_y> <kp3_v>
```

Ví dụ thực tế:

```
0 0.478516 0.491943 0.940755 0.537109 0.068359 0.289307 2 0.948893 0.223389 2 0.917969 0.760498 2 0.008138 0.754395 2
```

---

## Giải thích từng trường

### Class ID

```
0  ← class index, chỉ có 1 class duy nhất: "slide"
```

---

### Bounding Box: `cx cy bw bh`

Tất cả đều **normalized theo kích thước ảnh gốc** (giá trị trong `[0, 1]`).

| Trường | Ý nghĩa | Công thức |
|--------|---------|-----------|
| `cx` | x tâm bbox | `(x_min + x_max) / 2 / image_width` |
| `cy` | y tâm bbox | `(y_min + y_max) / 2 / image_height` |
| `bw` | chiều rộng bbox | `(x_max - x_min) / image_width` |
| `bh` | chiều cao bbox | `(y_max - y_min) / image_height` |

Bbox được tính tự động từ `min/max` của 4 keypoints — tức là bbox là hình chữ nhật bao ngoài 4 góc slide.

**Ví dụ** — ảnh 1280*720:
```
cx = 0.478516 → x_center = 0.478516 * 1280 ≈ 612 px từ trái
cy = 0.491943 → y_center = 0.491943 * 720  ≈ 354 px từ trên
bw = 0.940755 → width    = 0.940755 * 1280 ≈ 1204 px
bh = 0.537109 → height   = 0.537109 * 720  ≈ 387 px
```

---

### Keypoints: `kp_x kp_y kp_v` (×4)

Mỗi keypoint gồm 3 giá trị. Thứ tự cố định:

| Index | Tên | Mô tả |
|-------|-----|-------|
| kp0 | `top_left` | góc trên trái slide |
| kp1 | `top_right` | góc trên phải slide |
| kp2 | `bottom_right` | góc dưới phải slide |
| kp3 | `bottom_left` | góc dưới trái slide |

**Tọa độ `kp_x`, `kp_y`** — normalized theo ảnh gốc (không phải theo bbox):

```
kp_x = pixel_x / image_width
kp_y = pixel_y / image_height
```

**Visibility `kp_v`** — chuẩn COCO:

| Giá trị | Ý nghĩa |
|---------|---------|
| `0` | không tồn tại / không annotate |
| `1` | có nhưng bị che khuất |
| `2` | có và visible hoàn toàn |

**Ví dụ** — ảnh 1280*720:
```
kp0 top_left     = (0.068359, 0.289307, 2) → pixel (87,  208) — gần cạnh trái, 1/3 từ trên
kp1 top_right    = (0.948893, 0.223389, 2) → pixel (1215, 161) — gần cạnh phải, cao hơn kp0
kp2 bottom_right = (0.917969, 0.760498, 2) → pixel (1175, 548) — gần cạnh phải, phía dưới
kp3 bottom_left  = (0.008138, 0.754395, 2) → pixel (10,  543) — gần cạnh trái, phía dưới
```

**Tại sao keypoint tính theo ảnh, không theo bbox?**  
YOLO detect bbox trước → crop → resize. Nếu kp tính theo bbox thì phải re-normalize mỗi lần crop. Tính theo ảnh gốc → transform 1 lần duy nhất, đơn giản hơn.

---

### Minh họa layout

```
(0,0)─────────────────────────────── image_width
  │
  │     kp0 (0.068, 0.289) ─────── kp1 (0.948, 0.223)
  │          │                            │
  │          │        bbox center         │
  │          │       (0.478, 0.491)       │
  │          │                            │
  │     kp3 (0.008, 0.754) ─────── kp2 (0.917, 0.760)
  │
image_height

→ slide hơi nghiêng: kp0 thấp hơn kp1 (0.289 vs 0.223)
```

---

## Convert COCO → YOLO

Annotate trên Roboflow → export COCO JSON → chạy convert:

```bash
python convert_coco_to_yolo_pose.py <coco.json> <output_labels_dir>
```

Script sẽ:
1. Parse `images` và `annotations` từ JSON
2. Tính bbox từ min/max của 4 keypoints
3. Normalize tất cả về `[0, 1]` theo kích thước ảnh
4. Ghi ra file `.txt` cùng tên với ảnh
