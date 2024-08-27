"""Evaluation function to test the model performance."""

from typing import Tuple

import torch
import torch.nn as nn

def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion = None,
    ) -> Tuple[float, float]:
    
    """Validate the model on the entire test set."""
    if criterion is None: criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item() * target.size(0)
            _, predicted = torch.max(outputs.data, 1)  
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return loss, accuracy