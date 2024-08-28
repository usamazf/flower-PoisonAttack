"""A function to load the MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_emnist(data_root, download, split="mnist") -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        # transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.1307,))
    ])
    
    # Initialize Datasets. EMNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=data_root, train=True, split=split, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_root, train=False, split=split, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset
