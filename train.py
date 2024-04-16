from ultralytics import YOLO
from pathlib import Path
import argparse
import settings
import torch


def train():
    """
    Function to train the YOLOv8 model.
    """
    # Initialize model for training
    model_path = Path(settings.DETECTION_MODEL)
    model = YOLO(model_path)
    
    #freeze_until = "model.model.6" 
    #for name, parameter in model.named_parameters():
    #    if name.startswith(freeze_until):
    #        break
    #    parameter.requires_grad = False

    model.train(data='datasets/data.yaml', epochs=100, imgsz=640, resume=True)
    

if __name__ == "__main__":
    train()