# BGU SISE (2026 A) - ML Assignment 4

By -
1. Stav Asher - Data Preprocessing
2. Aaron Iziyaev - Train code, Factories & Configuration
3. Lior Jermin - Test & Validation

## Data

The models were fine-tuned / augmented with the following dataset per request:

- [ 102 Category Flower Dataset ](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Follow the code in `./dataset`, and add data under `./data` (ignored).
The image preprocessing converts the labels (`imagelabels.mat`) to csv format,
and extracts the images in `102flowers.tgz`.

## How to run
If on windows, run:
```shell
pip install -r requirements.txt
python main.py
```

If using linux or macos, use `python3` and `pip3` instead.
