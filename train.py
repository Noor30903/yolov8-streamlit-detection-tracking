from ultralytics import YOLO
from pathlib import Path
import argparse
import settings
def train():
    """
    Function to train the YOLOv8 model.
    """
    # Initialize model for training
    model_path = Path(settings.DETECTION_MODEL)
    model = YOLO(model_path)
    # Start training
    model.train(data='datasets/data.yaml', epochs=100, imgsz=640)
    model.save(model_path)

if __name__ == "__main__":
    train()