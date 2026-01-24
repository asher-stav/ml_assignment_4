"""
Will be changed later after a POC for fine tuning
"""

import torch
import numpy as np
import torchvision.models as models


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # set model to training mode
    run_loss, correct, total = 0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return run_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    pass


def build_vgg19():
    model = models.vgg19(pretrained=True)
    pass


def build_yolo_v5():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    pass