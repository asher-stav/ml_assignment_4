
import torch
from finetune_utils import log


def test_model(model, dataloader, criterion, device):

    model.eval()  # set model to evaluation mode
    run_loss, correct, total = 0, 0, 0

    log(f'Starting test on Model:     {model.__class__.__name__}, ')
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

    test_acc = correct / total
    test_loss = run_loss / len(dataloader)
    
    log(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
    
    return test_loss, test_acc