from ultralytics import YOLO
from pathlib import Path
import argparse
import settings
def train(data_path, model_output_path, epochs=100, img_size=640, batch_size=16):
    """
    Function to train the YOLOv8 model.
    """

    # Initialize model for training
    model_path = Path(settings.DETECTION_MODEL)
    model = YOLO(model_path)
    # Start training
    model.train(data='datasets/data.yaml', epochs=100, imgsz=640)
    model.save(model_path)