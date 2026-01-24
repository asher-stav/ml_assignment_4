import torch
import torchvision.models as models
from utils import log


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch
    Returns: epoch loss and epoch accuracy
    """
    model.train()  # set model to training mode
    run_loss, correct, total = 0, 0, 0
    
    log(f'Starting train on Model:     {model.__class__.__name__}, ')
    log(f'                  Optimizer: {optimizer.__class__.__name__}, ')
    log(f'                  Criterion: {criterion.__class__.__name__}')
    log(f'Using device: {device}')
    
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
        
    epoch_loss = run_loss / len(dataloader)
    epoch_acc = correct / total
    
    log(f'Epoch Loss: {epoch_loss:.4f} | Epoch Acc: {epoch_acc:.4f}')
        
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validates the model
    Returns: validation accuracy
    """
    model.eval()  # set model to evaluation mode
    correct, total = 0, 0

    log(f'Starting validate on Model:     {model.__class__.__name__}, ')
    log(f'                  Dataloader: {dataloader.__class__.__name__}, ')
    log(f'                  Criterion: {criterion.__class__.__name__}')
    log(f'Using device: {device}')

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    
    log(f'Validation Acc: {val_acc:.4f}')
    
    return val_acc
    

def build_vgg19(num_classes, device, free_features=False):    
    """
    Replaces the last layer of VGG19 & updates the device.
    Returns: model
    """
    model = models.vgg19(pretrained=True)
    
    if free_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    log(f'VGG19 Last layer replaced with:           {model.classifier[-1].__class__.__name__}')
    log(f'      Number of in features (last layer): {in_features}')
    log(f'      Number of classes:                  {num_classes}')
    
    model.to(device)
    
    return model


def build_yolo_v5(num_classes, device, free_features=False):
    """
    Replaces the last layer of YOLOv5 & updates the device.
    Returns: model
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    if free_features:
        for param in model.model.parameters():
            param.requires_grad = False
            
    in_features = model.model[-1].m.in_channels
    model.model[-1].m = torch.nn.Linear(in_features, num_classes)

    log(f'YOLOv5 Last layer replaced with:           {model.model[-1].m.__class__.__name__}')
    log(f'      Number of in features (last layer): {in_features}')
    log(f'      Number of classes:                  {num_classes}')

    model.to(device)
    
    return model
