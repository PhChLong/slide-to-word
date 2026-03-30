from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params

#?: Đổi sang yolov8n-pose để output keypoints thay vì bounding box
def load_model(pretrained: bool = False) -> YOLO:
    #?: yolov8n-pose.pt = pretrained COCO pose (17 keypoints người)
    #?: yolov8n-pose.yaml = architecture only, dùng khi fine-tune với data của mình
    model_name = "yolov8n-pose.pt" if pretrained else "yolov8n-pose.yaml"
    return YOLO(model_name)

def print_architecture(yolo_model: YOLO) -> None:
    model = yolo_model.model
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nModel structure:")
    print(model)

#?: Hàm mới: extract 4 corner từ keypoint output
def extract_corners(result) -> list[dict] | None:
    """
    Trả về list các slide được detect, mỗi slide có 4 corners.
    Thứ tự keypoint theo label: top_left, top_right, bottom_right, bottom_left
    """
    #?: result.keypoints.xy shape: (num_detections, num_keypoints, 2)
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    
    slides = []
    keypoints_xy = result.keypoints.xy        #?: tọa độ (x, y) pixel
    keypoints_conf = result.keypoints.conf    #?: confidence từng keypoint

    for i in range(len(keypoints_xy)):
        kp = keypoints_xy[i]      #?: shape (4, 2) — 4 corners
        cf = keypoints_conf[i]    #?: shape (4,)

        #?: Đảm bảo đủ 4 keypoints
        if kp.shape[0] < 4:
            continue

        slides.append({
            "top_left":     (float(kp[0][0]), float(kp[0][1])),
            "top_right":    (float(kp[1][0]), float(kp[1][1])),
            "bottom_right": (float(kp[2][0]), float(kp[2][1])),
            "bottom_left":  (float(kp[3][0]), float(kp[3][1])),
            "confidence":   float(result.boxes.conf[i]) if result.boxes is not None else None,
        })
    
    return slides if slides else None

if __name__ == "__main__":
    pretrained = True
    model = load_model(pretrained=pretrained)
    
    image_path = Path(__file__).resolve().parents[2] / "data" / "Messenger_creation_3B644284-1B1B-4274-ADF8-003649E00D97.jpeg"
    results = model.predict(source=str(image_path), verbose=False)
    result = results[0]

    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_yolov8n_pose.jpg"
    
    annotated_image = Image.fromarray(result.plot())
    annotated_image.save(output_path)
    print(f"Saved annotated image to: {output_path}")

    #?: In corner coordinates thay vì raw boxes
    corners = extract_corners(result)
    if corners:
        for idx, slide in enumerate(corners):
            print(f"\nSlide {idx + 1}:")
            for key, val in slide.items():
                print(f"  {key}: {val}")
    else:
        print("No slides detected.")
# ```

# ## Chuẩn bị data để fine-tune

# Khi bạn tự chuẩn bị label, format YOLO Pose yêu cầu mỗi dòng trong `.txt`:
# ```
# <class_id> <cx> <cy> <w> <h> <kp1_x> <kp1_y> <kp1_vis> <kp2_x> <kp2_y> <kp2_vis> <kp3_x> <kp3_y> <kp3_vis> <kp4_x> <kp4_y> <kp4_vis>