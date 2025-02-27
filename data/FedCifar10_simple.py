import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import platform
from collections import Counter
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter

def get_dataloaders(client_num, data_root, seed, param={
    'Batchsize': 16, 'dirichlet': 1, 'val': 0.1
}):
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    val_split = param['val']
    total_class = 10  # CIFAR-10 classes
    unknown_label = 11  # Assign SVHN data to class 11

    np.random.seed(seed)
    state = np.random.get_state()

    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 as known class dataset
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    trainx, trainy = np.array(train_dataset.data), np.array(train_dataset.targets)
    testx, testy = np.array(test_dataset.data), np.array(test_dataset.targets)

    # Load SVHN as unknown class dataset
    svhn_dataset = datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
    svhnx, svhny = svhn_dataset.data, np.full(svhn_dataset.data.shape[0], unknown_label)
    svhnx = np.transpose(svhnx, (0, 2, 3, 1))  # Convert to (N, H, W, C) format to match CIFAR-10

    # Merge CIFAR-10 test set with SVHN test set
    testx = np.concatenate((testx, svhnx), axis=0)
    testy = np.concatenate((testy, svhny), axis=0)

    # Create dataset indices for known classes
    train_data_known_idx = np.where(np.isin(trainy, range(total_class)))[0]
    test_data_known_idx = np.where(np.isin(testy, range(total_class)))[0]
    test_data_unknown_idx = np.where(testy == unknown_label)[0]

    # Validation split
    np.random.shuffle(train_data_known_idx)
    val_size = int(len(train_data_known_idx) * val_split)
    val_data_known_idx, train_data_known_idx = train_data_known_idx[:val_size], train_data_known_idx[val_size:]

    num_workers = 0 if platform.system() == 'Windows' else 4
    client_idcs = dirichlet_split_noniid(trainy[train_data_known_idx], alpha=dirichlet_alpha, n_clients=client_num, state=state)

    trainloaders = []
    for i in range(client_num):
        sub_train_idx = train_data_known_idx[client_idcs[i]]
        client_trainset = data.Subset(train_dataset, sub_train_idx)
        client_loader = data.DataLoader(client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
        trainloaders.append(client_loader)

    valloader = data.DataLoader(data.Subset(train_dataset, val_data_known_idx), batch_size=batchsize, shuffle=False, num_workers=num_workers)

    merged_test_dataset = CustomDataset(testx, testy, transform=transform)
    closeloader = data.DataLoader(data.Subset(merged_test_dataset, test_data_known_idx), batch_size=1, shuffle=False, num_workers=num_workers)
    openloader = data.DataLoader(data.Subset(merged_test_dataset, test_data_unknown_idx), batch_size=1, shuffle=False, num_workers=num_workers)


    return trainloaders, valloader, closeloader, openloader

class CustomDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx] 
        img = Image.fromarray(img) # Convert NumPy array to PIL image
        if self.transform:
            img = self.transform(img)
        return img, label


def dirichlet_split_noniid(train_labels, alpha, n_clients, state):
    n_classes = 8
    np.random.set_state(state)  # Ensure reproducibility
    total_classes = 8 # Number of unique classes

    # Get indices for each class
    class_indices = {c: np.where(train_labels == c)[0] for c in range(total_classes)}

    # Dirichlet-based sample distribution per class
    label_distribution = np.random.dirichlet([alpha] * n_clients, total_classes)  

    # Step 1: Assign samples to clients using Dirichlet
    client_indices = [[] for _ in range(n_clients)]
    client_classes = [set() for _ in range(n_clients)]

    for class_id, proportions in enumerate(label_distribution):
        class_idcs = class_indices[class_id]  # Indices for current class
        proportions = (np.cumsum(proportions)[:-1] * len(class_idcs)).astype(int)
        split_indices = np.split(class_idcs, proportions)

        for i, idcs in enumerate(split_indices):
            client_indices[i].extend(idcs)
            client_classes[i].add(class_id)

    # Step 2: missing/excess labels
    for i in range(n_clients):
        while len(client_classes[i]) < n_classes:
            # Get missing classes
            missing_classes = set(range(total_classes)) - client_classes[i]
            if not missing_classes:
                break 
            new_class = np.random.choice(list(missing_classes))  # Pick random missing class

            # Add random 5 samples from the missing class
            extra_samples = np.random.choice(class_indices[new_class], size=5, replace=False)
            client_indices[i].extend(extra_samples)
            client_classes[i].add(new_class)

        while len(client_classes[i]) > n_classes:
            # Remove excess class
            excess_class = np.random.choice(list(client_classes[i]))
            client_indices[i] = [idx for idx in client_indices[i] if train_labels[idx] != excess_class]
            client_classes[i].remove(excess_class)

        client_indices[i] = np.array(client_indices[i])

    return client_indices





