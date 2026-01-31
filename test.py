
import torch
from finetune_utils import log


def test_model(model, dataloader, criterion, device):

    model.eval()  # set model to evaluation mode
    correct, total = 0, 0

    log(f'Starting test on Model:     {model.__class__.__name__}, ')
    log(f'                  Dataloader: {dataloader.__class__.__name__}, ')
    log(f'                  Criterion: {criterion.__class__.__name__}')
    log(f'Using device: {device}')

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    test_acc = correct / total
    
    log(f'Test Acc: {test_acc:.4f}')
    
    return test_acc
