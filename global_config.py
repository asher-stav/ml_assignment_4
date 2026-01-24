"""
A config so we could change things in one place instead of everywhere in the code
"""

import torch

## ==============================================================================
## General configs
## ==============================================================================

__builders_config = {
    'num_of_classes': 10,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'freeze': False
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
    'debug': False,

    # i don't like the key name
    'builders_config': __builders_config,
    
    # Must have a 'name' key
    'optimizer': __adam_config,
    
    # Criterion name
    'criterion': 'CrossEntropyLoss'
}