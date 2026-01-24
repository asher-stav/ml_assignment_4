import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path
import tarfile
from sklearn.model_selection import train_test_split


data_path = Path('data')


def labels_to_csv():
    print('Converting labels to CSV format...')
    mat_path = data_path / 'imagelabels.mat'
    mat_data = scipy.io.loadmat(mat_path)

    print(mat_data.keys())

    labels = mat_data['labels'].flatten()

    df = pd.DataFrame({
        "image_id": np.arange(1, len(labels) + 1),
        "label": labels
    })

    csv_path = data_path / 'imagelabels.csv'
    df.to_csv(csv_path, index=False)
    print("CSV saved:", csv_path)


def extract_images():
    print('\nExtracting images...')
    images_archive = tarfile.open(data_path / '102flowers.tgz')
    images_archive.extractall(data_path)
    images_archive.close()


def split_train_val_test():
    print('\nSplitting data to train-validation-test...')
    df = pd.read_csv(data_path / 'imagelabels.csv')
    temp_df, test_df = train_test_split(df, test_size=0.25)
    train_df, val_df = train_test_split(temp_df, test_size=(1 / 3))

    train_df.to_csv(data_path / 'trainlabels.csv', index=False)
    val_df.to_csv(data_path / 'vallabels.csv', index=False)
    test_df.to_csv(data_path / 'testlabels.csv', index=False)


def main():
    global data_path
    data_path = Path('../data/')
    labels_to_csv()
    extract_images()
    split_train_val_test()


if __name__ == '__main__':
    main()
