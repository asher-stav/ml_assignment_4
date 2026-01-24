"""
Factory functions for criterion and optimizer to be produced according to the configuration
"""

import torch
from global_config import CONFIG as cfg
from utils import log

def criterion_factory():
    criterion_name = cfg['criterion']
    
    log(f'Using criterion: {criterion_name}')
    
    # Add more criterions here
    if criterion_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    
    log(f'Criterion: {criterion_name} is configured in the config file, but is not implemented in the criterion factory')
    
    raise ValueError(f'Unknown criterion: {criterion_name}')


def optimizer_factory(model):
    optimizer_cfg = cfg['optimizer']
    optimizer_name = optimizer_cfg['name']
    
    log(f'Using optimizer: {optimizer_name}')
    
    # Add more optimizers here
    if optimizer_name == 'Adam':
        lr = optimizer_cfg['lr']
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr)
        
    log(f'Optimizer: {optimizer_name} is configured in the config file, but is not implemented in the optimizer factory')
    
    raise ValueError(f'Unknown optimizer: {optimizer_name}')