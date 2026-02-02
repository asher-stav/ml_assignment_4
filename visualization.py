import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import pickle

# needed for some reason
matplotlib.use('Agg') 
__result_dir = "results"

def value_graph_over_epochs(model_name, value, data: List[Tuple[float, float, float]]):
    """
    data has entries per epoch of:
    1. Train
    2. Validation
    3. Test
    """
    epochs = list(range(1, len(data) + 1))
    train_vals = [t[0] for t in data]
    val_vals = [t[1] for t in data]
    test_vals = [t[2] for t in data]
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(epochs, train_vals, label='Train', marker='o')
    ax.plot(epochs, val_vals, label='Validation', marker='o')
    ax.plot(epochs, test_vals, label='Test', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value)
    ax.set_title(f'{value} over Epochs for {model_name}')
    ax.grid(True)
    ax.legend()
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_{value}_over_epochs_{timestamp}.png"
    __save_file(fig, file_name.replace(" ", "_"))

    
def __save_file(fig, file_name):
    os.makedirs(__result_dir, exist_ok=True)
    fig.savefig(os.path.join(__result_dir, file_name), dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)


def main():
    with open('results/vgg19.pkl', 'rb') as f:
        vgg19_results = pickle.load(f)
    
    train_loss = vgg19_results['train_loss']
    validation_loss = vgg19_results['validation_loss']
    test_loss = vgg19_results['test_loss']
    value_graph_over_epochs("VGG19", "Loss", list(zip(train_loss, validation_loss, test_loss)))

    train_accuracies = vgg19_results['train_accuracies']
    validation_accuracies = vgg19_results['validation_accuracies']
    test_accuracies = vgg19_results['test_accuracies']
    value_graph_over_epochs("VGG19", "Accuracy", list(zip(train_accuracies, validation_accuracies, test_accuracies)))
    
    
    with open('results/yolov5.pkl', 'rb') as f:
        yolov5_results = pickle.load(f)
    
    train_loss = yolov5_results['train_loss']
    validation_loss = yolov5_results['validation_loss']
    test_loss = yolov5_results['test_loss']
    value_graph_over_epochs("YOLOv5", "Loss", list(zip(train_loss, validation_loss, test_loss)))

    train_accuracies = yolov5_results['train_accuracies']
    validation_accuracies = yolov5_results['validation_accuracies']
    test_accuracies = yolov5_results['test_accuracies']
    value_graph_over_epochs("YOLOv5", "Accuracy", list(zip(train_accuracies, validation_accuracies, test_accuracies)))


if __name__ == "__main__":
    main()
