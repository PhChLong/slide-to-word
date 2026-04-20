from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s-pose.pt")
    results = model.train(data="D:/Github Repos/slide-to-word/dataset/data.yaml", epochs=100, imgsz=640, device = 0, workers = 0)

# from pathlib import Path
# data_yaml = Path("D:/Github Repos/slide-to-word/dataset/data.yaml")
# print(data_yaml.exists())