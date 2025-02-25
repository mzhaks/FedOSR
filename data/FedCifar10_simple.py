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

# def get_dataloaders(client_num, data_root, seed, param={
#     'Known_class': 7, 'unKnown_class': 3, 'Batchsize': 16, 'dirichlet': 1, 'val' : 0.1
# }):
#     known_class, unknown_class = param['Known_class'], param['unKnown_class']
#     batch_size, dirichlet_alpha = param['Batchsize'], param['dirichlet']
#     total_class = 10
#     assert known_class + unknown_class <= total_class
    
#     np.random.seed(seed)
#     class_list = np.random.permutation(total_class)
#     known_classes, unknown_classes = class_list[:known_class], class_list[known_class:]
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
#     test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
#     train_labels, test_labels = np.array(train_set.targets), np.array(test_set.targets)
    
#     class_map = {cls: i for i, cls in enumerate(known_classes)}
#     unknown_map = {cls: i + known_class for i, cls in enumerate(unknown_classes)}
#     label_map = {**class_map, **unknown_map}
    
#     train_labels_mapped = np.vectorize(label_map.get, otypes=[int])(train_labels, train_labels)
#     test_labels_mapped = np.vectorize(label_map.get, otypes=[int])(test_labels, test_labels)
    
#     train_indices = {cls: np.where(train_labels == cls)[0] for cls in known_classes}
#     test_known_indices = np.hstack([np.where(test_labels == cls)[0] for cls in known_classes])
#     test_unknown_indices = np.hstack([np.where(test_labels == cls)[0] for cls in unknown_classes])
    
#     all_train_indices = np.hstack(list(train_indices.values()))
#     client_indices = dirichlet_split_noniid(train_labels_mapped[all_train_indices], alpha=dirichlet_alpha, n_clients=client_num, state=np.random.get_state())
    
#     train_loaders = [DataLoader(Subset(train_set, client_indices[i]), batch_size=batch_size, shuffle=True, num_workers=4) for i in range(client_num)]
    
#     val_loader = DataLoader(Subset(test_set, test_known_indices), batch_size=1, shuffle=False, num_workers=4)
#     close_loader = DataLoader(Subset(test_set, test_known_indices), batch_size=1, shuffle=False, num_workers=4)
#     open_loader = DataLoader(Subset(test_set, test_unknown_indices), batch_size=1, shuffle=False, num_workers=4)
    
#     return train_loaders, val_loader, close_loader, open_loader


###############original################
# def get_dataloaders(client_num, data_root, seed, param={
#     'Known_class': 8, 'unKnown_class': 2, 'Batchsize': 16, 'dirichlet': 1, 'val' : 0.1
# }):
#     known_class = param['Known_class']
#     unknown_class = param['unKnown_class']
#     batchsize = param['Batchsize']
#     dirichlet_alpha = param['dirichlet']
#     val_split = param['val']
#     total_class = 10
#     assert known_class + unknown_class <= total_class
    
#     np.random.seed(seed)
#     state = np.random.get_state()
#     class_list = np.arange(total_class)
#     np.random.shuffle(class_list)
#     known_class_list = class_list[:known_class]
#     unknown_class_list = class_list[known_class:known_class + unknown_class]
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
#     test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
#     trainx, trainy = np.array(train_dataset.data), np.array(train_dataset.targets)
#     testx, testy = np.array(test_dataset.data), np.array(test_dataset.targets)
    
#     # Dict key value pair (key,value) == (original_idx, new_idx)
#     known_dict = {c: i for i, c in enumerate(known_class_list)}
#     unknown_dict = {c: i + len(known_class_list) for i, c in enumerate(unknown_class_list)}
    
#     # relabel the original dataset 
#     trainy = np.array([known_dict[y] if y in known_dict else unknown_dict[y] for y in trainy])
#     testy = np.array([known_dict[y] if y in known_dict else unknown_dict[y] for y in testy])
    
#     train_data_known_idx = np.where(np.isin(trainy, range(known_class)))[0]
#     test_data_known_idx = np.where(np.isin(testy, range(known_class)))[0]
#     test_data_unknown_idx = np.setdiff1d(np.arange(len(testy)), test_data_known_idx)

#     print("train_data_known_idx",train_data_known_idx )

#     #val
#     np.random.shuffle(train_data_known_idx)

#     val_size = int(len(train_data_known_idx) * val_split)
#     val_data_known_idx, train_data_known_idx = train_data_known_idx[:val_size], train_data_known_idx[val_size:]
    
#     num_workers = 0 if platform.system() == 'Windows' else 4
#     client_idcs = dirichlet_split_noniid(trainy[train_data_known_idx], alpha=dirichlet_alpha, n_clients=client_num, state=state)
    
#     trainloaders = []
#     unique__all_labels = []
#     for i in range(client_num):
#         sub_train_idx = train_data_known_idx[client_idcs[i]]
#         client_trainset = data.Subset(train_dataset, sub_train_idx)
#         # Extract all labels for this client
#         all_labels = [train_dataset.targets[idx] for idx in sub_train_idx]
#         unique_labels = np.unique(all_labels)

#         label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

#         class LabelMappedDataset(data.Dataset):
#             def __init__(self, subset, label_map):
#                 self.subset = subset
#                 self.label_map = label_map

#             def __len__(self):
#                 return len(self.subset)

#             def __getitem__(self, idx):
#                 image, label = self.subset[idx]  
#                 return image, self.label_map[label]  

#         mapped_client_trainset = LabelMappedDataset(client_trainset, label_map)

#         client_loader = data.DataLoader(mapped_client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    
#         # client_loader = data.DataLoader(client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
#         all_labels = []
#         for images, labels in client_loader:
#             all_labels.extend(labels.numpy())

#         unique_labels = np.unique(all_labels)
#         print(unique_labels)
#         trainloaders.append(client_loader)
#         # unique__all_labels.append(unique_labels)
    
#     valloader = data.DataLoader(data.Subset(train_dataset, val_data_known_idx), batch_size=batchsize, shuffle=False, num_workers=num_workers)
#     closeloader = data.DataLoader(data.Subset(test_dataset, test_data_known_idx), batch_size=1, shuffle=False, num_workers=num_workers)
#     openloader = data.DataLoader(data.Subset(test_dataset, test_data_unknown_idx), batch_size=1, shuffle=False, num_workers=num_workers)
    
#     return trainloaders, valloader, closeloader, openloader

#######################################################################

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

    # Now use this merged dataset for the loaders
    closeloader = data.DataLoader(data.Subset(merged_test_dataset, test_data_known_idx), batch_size=1, shuffle=False, num_workers=num_workers)
    openloader = data.DataLoader(data.Subset(merged_test_dataset, test_data_unknown_idx), batch_size=1, shuffle=False, num_workers=num_workers)
    # closeloader = data.DataLoader(data.Subset(test_dataset, test_data_known_idx), batch_size=1, shuffle=False, num_workers=num_workers)
    # openloader = data.DataLoader(data.Subset(test_dataset, test_data_unknown_idx), batch_size=1, shuffle=False, num_workers=num_workers)

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
    np.random.set_state(state)
    label_distribution = np.random.dirichlet([alpha] * n_clients, len(np.unique(train_labels)))
    client_idcs = [[] for _ in range(n_clients)]
    
    for k, idcs in enumerate(np.argsort(train_labels)):
        proportions = label_distribution[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idcs)).astype(int)[:-1]
        client_idcs = [np.concatenate([cid, idcs[start:end]]) for cid, start, end in zip(client_idcs, [0] + list(proportions), list(proportions) + [len(idcs)])]
    
    return [list(map(int, idcs)) for idcs in client_idcs]


#######################################################################################
# def get_dataloaders(client_num, data_root, seed, param={
#     'Known_class': 8, 'unKnown_class': 2, 'Batchsize': 8, 'dirichlet': 1.0}):
    
#     known_class = param['Known_class']
#     unknown_class = param['unKnown_class']
#     batchsize = param['Batchsize']
#     dirichlet_alpha = param['dirichlet']
#     total_class = 10  # CIFAR-10 has 10 classes
    
#     assert known_class + unknown_class <= total_class
    
#     np.random.seed(seed)
#     state = np.random.get_state()
#     class_list = np.arange(total_class)
#     np.random.shuffle(class_list)
#     known_class_list = class_list[:known_class]
#     unknown_class_list = class_list[known_class:known_class + unknown_class]
    
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     full_trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
#     full_testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
#     train_labels = np.array(full_trainset.targets)
#     test_labels = np.array(full_testset.targets)
    
#     known_dict = {cls: i for i, cls in enumerate(known_class_list)}
#     unknown_dict = {cls: i + len(known_class_list) for i, cls in enumerate(unknown_class_list)}
    
#     train_labels_mapped = np.copy(train_labels)
#     test_labels_mapped = np.copy(test_labels)
    
#     for cls in known_class_list:
#         train_labels_mapped[train_labels == cls] = known_dict[cls]
#         test_labels_mapped[test_labels == cls] = known_dict[cls]
#     for cls in unknown_class_list:
#         train_labels_mapped[train_labels == cls] = unknown_dict[cls]
#         test_labels_mapped[test_labels == cls] = unknown_dict[cls]
    
#     train_known_idx = np.where(train_labels_mapped < known_class)[0]
#     test_known_idx = np.where(test_labels_mapped < known_class)[0]
#     test_unknown_idx = np.where(test_labels_mapped >= known_class)[0]
    
#     print(f'Known: {len(train_known_idx)} samples, Unknown: {len(test_unknown_idx)} samples')
    
#     client_idcs = dirichlet_split_noniid(train_labels_mapped[train_known_idx], alpha=dirichlet_alpha, n_clients=client_num, state=state)
    
#     trainloaders = []
#     for i in range(client_num):
#         sub_trainset = Subset(full_trainset, train_known_idx[client_idcs[i]])
#         trainloaders.append(DataLoader(sub_trainset, batch_size=batchsize, shuffle=True, num_workers=4 if platform.system() != 'Windows' else 0))
    
#     valloader = DataLoader(Subset(full_trainset, train_known_idx), batch_size=batchsize, shuffle=False, num_workers=4)
#     closeloader = DataLoader(Subset(full_testset, test_known_idx), batch_size=batchsize, shuffle=False, num_workers=4)
#     openloader = DataLoader(Subset(full_testset, test_unknown_idx), batch_size=batchsize, shuffle=False, num_workers=4)
    
#     return trainloaders, valloader, closeloader, openloader

# def dirichlet_split_noniid(train_labels, alpha, n_clients, state):
#     """
#     Perform non-IID data partitioning using a Dirichlet distribution.

#     Args:
#         train_labels (np.array): Array of labels for the training set.
#         alpha (float): Dirichlet distribution concentration parameter.
#         n_clients (int): Number of clients for data partitioning.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         list: A list of `n_clients` arrays containing assigned sample indices.
#     """
#     np.random.set_state(state)  # Ensure reproducibility

#     n_classes = train_labels.max() + 1  # Number of unique classes
#     label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)  
#     # (n_classes, n_clients) matrix where each row sums to 1.

#     # Get sample indices for each class
#     class_indices = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

#     # Initialize client index lists
#     client_indices = [[] for _ in range(n_clients)]

#     # Assign samples to clients based on Dirichlet proportions
#     for class_idcs, proportions in zip(class_indices, label_distribution):
#         proportions = (np.cumsum(proportions)[:-1] * len(class_idcs)).astype(int)
#         split_indices = np.split(class_idcs, proportions)  # Split indices per client

#         for i, idcs in enumerate(split_indices):
#             client_indices[i].append(idcs)

#     # Concatenate all assigned indices per client
#     client_indices = [np.concatenate(idcs) for idcs in client_indices]

#     return client_indices
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

    # Step 2: Fix missing/excess labels
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
# def dirichlet_split_noniid(labels, alpha, n_clients, state):
#     label_distribution = np.random.dirichlet([alpha] * n_clients, len(np.unique(labels)))
#     class_idcs = [np.where(labels == y)[0] for y in np.unique(labels)]
#     client_idcs = [[] for _ in range(n_clients)]
    
#     for c, fracs in zip(class_idcs, label_distribution):
#         np.random.set_state(state)
#         np.random.shuffle(c)
#         split_points = (np.cumsum(fracs) * len(c)).astype(int)[:-1]
#         splits = np.split(c, split_points)
#         for i, split in enumerate(splits):
#             client_idcs[i].extend(split)
    
#     return client_idcs




