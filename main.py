from torch.utils.data import DataLoader
from tqdm import tqdm

from global_config import CONFIG as cfg
from utils import log
import dataset.preprocessing as preprocessing
import dataset.image_dataset as image_dataset
import factories
import finetune


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


def train_model(model, transform):
    builders_cfg = cfg['builders_config']
    device = builders_cfg['device']
    batch_size = builders_cfg['batch_size']
    
    train_loader, val_loader, test_loader = get_dataloaders(transform, batch_size)
    log('Created DataLoaders')

    optimizer = factories.optimizer_factory(model)
    criterion = factories.criterion_factory()

    log('Training...')
    epochs = builders_cfg['epochs']
    for _ in tqdm(range(epochs)):
        finetune.train_epoch(model, train_loader, optimizer, criterion, device)
        finetune.validate(model, val_loader, criterion, device)

def main():
    # preprocessing.labels_to_csv()
    # preprocessing.extract_images()
    # preprocessing.split_train_val_test()

    builders_cfg = cfg['builders_config']

    num_classes = builders_cfg['num_of_classes']
    device = builders_cfg['device']
    freeze = builders_cfg['freeze']
    vgg19 = finetune.build_vgg19(num_classes, device, free_features=freeze)
    transform = image_dataset.vgg19_transform
    
    train_model(vgg19, transform)


if __name__ == '__main__':
    main()
