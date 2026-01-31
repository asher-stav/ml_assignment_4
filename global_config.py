"""
A config so we could change things in one place instead of everywhere in the code
"""

import torch

## ==============================================================================
## General configs
## ==============================================================================

__builders_config = {
    'num_of_classes': 102,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'freeze': False,
    'epochs': 16,
    'batch_size': 32
}

## ==============================================================================
## Optimizer Configs
## ==============================================================================

__adam_config = {
    'name': 'Adam',
    'lr': 0.001,
}


## ==============================================================================
## Exported Config
## ==============================================================================

CONFIG = {
    # determines if logs will be printed
    'debug': True,

    # i don't like the key name
    'builders_config': __builders_config,
    
    # Must have a 'name' key
    'optimizer': __adam_config,
    
    # Criterion name
    'criterion': 'CrossEntropyLoss',

    'random_seed': 1
}
