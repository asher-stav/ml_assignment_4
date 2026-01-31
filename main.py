from torch.utils.data import DataLoader
import torch
import pickle
from pathlib import Path
import numpy as np
import random

from global_config import CONFIG as cfg
from finetune_utils import log
import dataset.preprocessing as preprocessing
import dataset.image_dataset as image_dataset
import factories
import finetune
import test


def get_dataloaders(transform, batch_size):
    train_dataset = image_dataset.ImagesDataset(image_dataset.TRAIN_LABELS_PATH,
                                                transform=transform)
    val_dataset = image_dataset.ImagesDataset(image_dataset.VAL_LABELS_PATH,
                                                transform=transform)
    test_dataset = image_dataset.ImagesDataset(image_dataset.TEST_LABELS_PATH,
                                                transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def evaluate_model(model, transform, criterion):
    builders_cfg = cfg['builders_config']
    device = builders_cfg['device']
    batch_size = builders_cfg['batch_size']

    train_loader, val_loader, test_loader = get_dataloaders(transform, batch_size)
    log('Created DataLoaders')

    optimizer = factories.optimizer_factory(model)
    
    train_loss = []
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    log('Training...')
    epochs = builders_cfg['epochs']

    for _ in range(epochs):
        epoch_loss, train_acc = finetune.train_epoch(model, train_loader, optimizer, criterion, device)
        train_loss.append(epoch_loss)
        train_accuracies.append(train_acc)

        val_acc = finetune.validate(model, val_loader, criterion, device)
        validation_accuracies.append(val_acc)

        test_accuracy = test.test_model(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
    
    return train_loss, train_accuracies, validation_accuracies, test_accuracies


def save_results(results, filename):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / filename, 'wb') as file:
        pickle.dump(results, file)


def evaluate_vgg19(num_classes, device, freeze):
    log(f'\n--------  Evaluating VGG19 -------- ')
    vgg19 = finetune.build_vgg19(num_classes, device, free_features=freeze)
    transform = image_dataset.vgg19_transform

    criterion = factories.criterion_factory()
    
    train_loss, train_accuracies, validation_accuracies, test_accuracies = evaluate_model(vgg19, transform, criterion)
    results = {
        'train_loss': train_loss,
        'train_accuracies': train_accuracies,
        'validation_accuracies': validation_accuracies,
        'test_accuracies': test_accuracies
    }

    save_results(results, 'vgg19.pkl')


def evaluate_yolov5(num_classes, device, freeze):
    log('-------- Evaluating YOLOv5 --------')
    yolov5 = finetune.build_yolo_v5(num_classes, device, free_features=freeze)
    transform = image_dataset.yolov5_transform

    criterion = factories.criterion_factory()
    
    train_loss, train_accuracies, validation_accuracies, test_accuracies = evaluate_model(yolov5, transform, criterion)
    results = {
        'train_loss': train_loss,
        'train_accuracies': train_accuracies,
        'validation_accuracies': validation_accuracies,
        'test_accuracies': test_accuracies
    }

    save_results(results, 'yolov5.pkl')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    preprocessing.labels_to_csv()
    preprocessing.extract_images()
    preprocessing.split_train_val_test()

    seed = cfg['random_seed']
    set_random_seed(seed)


    builders_cfg = cfg['builders_config']

    num_classes = builders_cfg['num_of_classes']
    device = builders_cfg['device']
    freeze = builders_cfg['freeze']

    evaluate_vgg19(num_classes, device, freeze)
    evaluate_yolov5(num_classes, device, freeze)


if __name__ == '__main__':
    main()
