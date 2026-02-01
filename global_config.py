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
    'freeze': True,
    'epochs': 10,
    'batch_size': 32
}

## ==============================================================================
## Optimizer Configs
## ==============================================================================

__adam_config = {
    'name': 'Adam',
    'lr': 3e-4,
    'decay': 1e-4,
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
