import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path
import tarfile

data_path = Path('../data')

# Converting labels to csv


# Extracting images



def labels_to_csv():
    print('Converting labels to CSV format...')
    mat_path = data_path / 'imagelabels.mat'
    mat_data = scipy.io.loadmat(mat_path)

    print(mat_data.keys())

    labels = mat_data['labels'].flatten()

    df = pd.DataFrame({
        "image_id": np.arange(len(labels)),
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


def main():
    labels_to_csv()
    extract_images()

if __name__ == '__main__':
    main()
