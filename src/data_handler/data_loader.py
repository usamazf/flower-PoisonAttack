"""A function to load and split the desired dataset among clients."""

from .data_split import CustomDataset, split_data

def load_data(dataset_name: str, 
              dataset_path: str, 
              dataset_down: bool):
    
    assert dataset_name in ["MNIST", "CIFAR-10", "EMNIST"], f"Invalid dataset {dataset_name} requested."

    if dataset_name == "MNIST":
        from .dt_mnist import load_mnist
        trainset, testset = load_mnist(data_root=dataset_path, download=dataset_down)
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "EMNIST-DIGITS":
        from .dt_emnist import load_emnist
        trainset, testset = load_emnist(data_root=dataset_path, download=dataset_down, split="digits")
        # Modify data to add extra channel dimension
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "CIFAR-10":
        from .dt_cifar10 import load_cifar10
        trainset, testset = load_cifar10(data_root=dataset_path, download=dataset_down)
        # Modify data to have [S, C, H, W] format
        custom_trainset = CustomDataset(data=trainset.data.transpose((0, 3, 1, 2))/255.0, targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.transpose((0, 3, 1, 2))/255.0, targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "Fashion-MNIST":
        # Load Fashion-MNIST dataset
        from .dt_fmnist import load_fmnist
        trainset, testset = load_fmnist(data_root=dataset_path, download=dataset_down)
        # Modify data to add extra channel dimension
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "STL-10":
        # Load STL-10 dataset
        from .dt_stl10 import load_stl10
        trainset, testset = load_stl10(out_dir=dataset_path, download=dataset_down)
        # For some weird reason STL-10 has labels instead 
        # of targets adding additional attribute targets 
        # to make it consistent with other datasets
        custom_trainset = CustomDataset(data=trainset.data, targets=trainset.labels, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data, targets=testset.labels, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset    
    else:
        raise Exception(f"Invalid dataset {dataset_name} requested.")


def load_and_fetch_split(
        client_id: int,
        n_clients: int,
        dataset_conf: dict
    ):
    """A routine to load and split data."""

    # load the dataset requested
    trainset, testset \
        = load_data(dataset_name=dataset_conf["DATASET_NAME"],
                    dataset_path=dataset_conf["DATASET_PATH"],
                    dataset_down=dataset_conf["DATASET_DOWN"]
                   )

    # split the dataset if requested
    if dataset_conf["SPLIT"]:
        split_train, split_labels \
            = split_data(train_data = trainset, 
                         dirichlet_alpha = dataset_conf["DIRICHLET_ALPHA"], 
                         client_id = client_id,
                         n_clients = n_clients,
                         random_seed = dataset_conf["RANDOM_SEED"], 
                         worker_data = dataset_conf["WORKER_DATA"],
                         classes_per_worker = dataset_conf["CLASSES_PER_WORKER"]
                        )
        return (split_train, split_labels), testset
    else:
        return (trainset, None), testset