import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

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


def loss_graph_over_epochs(model_name, loss):
    epochs = list(range(1, len(loss) + 1))
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(epochs, loss, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Train Loss over Epochs for {model_name}')
    ax.grid(True)
    ax.legend()
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_loss_over_epochs_{timestamp}.png"
    __save_file(fig, file_name.replace(" ", "_"))

    
def __save_file(fig, file_name):
    os.makedirs(__result_dir, exist_ok=True)
    fig.savefig(os.path.join(__result_dir, file_name), dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)
