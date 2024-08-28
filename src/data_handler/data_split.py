"""Module containing functions for data spliting among clients."""

from typing import Callable, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset

class IdxSubset(torch.utils.data.Dataset):
    """Class to create custom subsets with indexes appended."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        #return (idx, *self.dataset[self.indices[idx]])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CustomDataset(Dataset):
    r"""
    Create a dataset with given data and labels

    Arguments:
        data (data): The data samples of the desired dataset.
        labels(sequence) : The respective labels of the provided data samples. 
    """
    def __init__(
            self,
            data: Union[list, np.ndarray, torch.Tensor], 
            targets: Union[list, np.ndarray, torch.Tensor], 
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ):
        self.data = data
        if not torch.is_tensor(self.data):
            self.data = torch.tensor(self.data)
        self.data = self.data.float()

        self.targets = targets
        if not torch.is_tensor(self.targets):
            self.targets = torch.tensor(self.targets)
        self.targets = self.targets.long()
        
        # original labels
        self.oTargets = self.targets.detach().clone()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.targets)

class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. 
                                will be the same length as indices
    """
    def __init__(self, dataset, indices, labels=None):
        self.dataset = dataset
        self.indices = indices

        # Extract data from provided dataset
        self.data = self.dataset[self.indices][0]
        
        # Setup data targets
        if labels is None:
            if hasattr(self.dataset, 'targets'):
                targets = np.array(self.dataset.targets)[indices]
            elif hasattr(self.dataset, 'labels'):
                targets = np.array(self.dataset.labels)[indices]
            else:
                # no targets or labels attribute in
                # the given dataset raise exception
                raise Exception("Dataset has no attribute targets or labels")
            self.targets = torch.tensor(targets.copy()).long()
        else:
            self.targets = torch.tensor(labels).long()

        # original labels
        self.oTargets = self.targets.detach().clone()
   
    def __getitem__(self, idx):
        #data = self.dataset[self.indices[idx]][0]
        data = self.data[idx]
        target = self.targets[idx]
        return (data, target)

    def __len__(self):
        return len(self.targets)
    
    def setTargets(self, labels):
        self.targets = torch.tensor(labels).long()   

def split_with_replacement(labels, n_workers, n_data, classes_per_worker):
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels) + 1
    # get label indcs
    label_idcs = {l : np.random.permutation(
        np.argwhere(np.array(labels)==l).flatten()
        ).tolist() for l in range(n_classes) }
    
    classes_per_worker = n_classes if classes_per_worker == 0 else classes_per_worker

    idcs = []
    for i in range(n_workers):
        worker_idcs = []
        budget = n_data
        c = np.random.randint(n_classes)
        while budget > 0:
            take = min(n_data // classes_per_worker, budget)
            worker_idcs.append(np.random.choice(label_idcs[c], take))
            budget -= take
            c = (c + 1) % n_classes
        idcs += [np.hstack(worker_idcs)]
    
    # print_split(idcs, labels)
    
    return idcs

def uneven_split(labels, n_workers, n_data, classes_per_worker):
    """Function to make uneven splits of given dataset for each worker as defined."""
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels) + 1
    # get label indcs
    label_idcs = {l : np.random.permutation(
        np.argwhere(np.array(labels)==l).flatten()
        ).tolist() for l in range(n_classes) }
    
    classes_per_worker = n_classes if classes_per_worker == 0 else classes_per_worker

    idcs = []
    for i in range(n_workers):
        worker_idcs = []
        budget = n_data[i]
        c = np.random.randint(n_classes)
        while budget > 0:
            take = min(n_data[i] // classes_per_worker, len(label_idcs[c]), budget)
            
            worker_idcs += label_idcs[c][:take]
            label_idcs[c] = label_idcs[c][take:]
            
            budget -= take
            c = (c + 1) % n_classes
        idcs += [worker_idcs]

    
    print_split(idcs, labels)
    
    return idcs


def split_dirichlet(labels, n_workers, alpha, double_stochstic=True):
    """Splits data among the workers using dirichlet distribution"""

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels)+1
    
    # get label distibution
    label_distribution = np.random.dirichlet([alpha]*n_workers, n_classes)
   
    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() for y in range(n_classes)]
    
    worker_idcs = [[] for _ in range(n_workers)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            worker_idcs[i] += [idcs]

    worker_idcs = [np.concatenate(idcs) for idcs in worker_idcs]

    # print_split(worker_idcs, labels)
  
    return worker_idcs

def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x

def split_data(
        train_data, 
        dirichlet_alpha,
        client_id, 
        n_clients, 
        random_seed = 32,
        worker_data = None, 
        classes_per_worker = None,
    ):
    """Split data among Worker nodes."""
    
    # Set random seed for reproducable results
    np.random.seed(random_seed)

    # Find allocated indices using dirichlet split
    if not worker_data:
        subset_idx = split_dirichlet(train_data.targets, n_clients, dirichlet_alpha)
    else:
        # if a list is not provided we
        # want to split data in equal chunks
        # of 
        if isinstance(worker_data, int):
            subset_idx = split_with_replacement(train_data.targets, n_clients, worker_data, classes_per_worker)
        else:
            subset_idx = uneven_split(train_data.targets, n_clients, worker_data, classes_per_worker)
    
    # Compute labels per worker
    label_counts = [np.bincount(np.array(train_data.targets)[i], minlength=10) for i in subset_idx]
    
    # Get actual worker data
    # worker_data = [IdxSubset(train_data, subset_idx[i]) for i in range(n_clients)]
    # print(f"Current Worker Split: {label_counts[client_id]}\n")
    worker_data = CustomSubset(train_data, subset_idx[client_id])

    # Return worker data splits
    return worker_data, label_counts[client_id]


def print_split(idcs, labels):
    """Helper function to print data splits made for workers."""
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Worker {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    
    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()