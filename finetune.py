import torch
import torchvision.models as models
from finetune_utils import log


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
        labels = labels.long()
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
    Returns: validation loss and validation accuracy
    """
    model.eval()  # set model to evaluation mode
    run_loss, correct, total = 0, 0, 0

    log(f'Starting validate on Model:     {model.__class__.__name__}, ')
    log(f'                  Dataloader: {dataloader.__class__.__name__}, ')
    log(f'                  Criterion: {criterion.__class__.__name__}')
    log(f'Using device: {device}')

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    val_loss = run_loss / len(dataloader)
    
    log(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}')

    return val_loss, val_acc
    

def build_vgg19(num_classes, device, free_features=True):    
    """
    Replaces the last layer of VGG19 & updates the device.
    Returns: model
    """
    model = models.vgg19(pretrained=True)
    
    if free_features:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    log(f'VGG19 Last layer replaced with:           {model.classifier[-1].__class__.__name__}')
    log(f'      Number of in features (last layer): {in_features}')
    log(f'      Number of classes:                  {num_classes}')
    
    model.to(device)
    
    return model


class YOLOv5Classifier(torch.nn.Module):
    def __init__(self, device, num_classes, free_features):
        super().__init__()
        self.device = device
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                           pretrained=True, autoshape=False)
        self.yolo.to(device)
        self.yolo.eval()

        self.free_features = free_features
    
        if free_features:
            for param in self.yolo.model.parameters():
                param.requires_grad = False

        # Hook pre-last layer to get its features before the detection
        self.features = None
        def hook(module, input, output):
            self.features = output
        self.yolo.model.model[-2].register_forward_hook(hook)

        # Create pool to prepare features for classification
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        
        in_features = 512
        self.classifier = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        if self.free_features:
            with torch.no_grad():
                _ = self.yolo(x)
        else:
            _ = self.yolo(x)
        x = self.features
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x
    
    def train(self, mode=True):
        super().train(mode)
        if self.free_features:
            self.yolo.eval()
        return self


def build_yolo_v5(num_classes, device, free_features=True):
    """
    Replaces the last layer of YOLOv5 & updates the device.
    Returns: model
    """
    model = YOLOv5Classifier(device, num_classes, free_features)

    log(f'YOLOv5 created with {num_classes} Classes')
    model.to(device)
    
    return model
