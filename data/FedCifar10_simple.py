import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import   random_split, ConcatDataset , Dataset
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

def get_dataloaders(client_num, data_root, seed, args, param={
    'Batchsize': 16, 'dirichlet': 1, 'val': 0.1}):
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    val_split = param['val']
    
    

    np.random.seed(seed)
    state = np.random.get_state()

    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.setting == 1 :
        # Load CIFAR-10 as known class dataset and unknown class dataset
        total_class = args.known_class  # CIFAR-10 classes
        unknown_label = args.known_class  # Assign SVHN data to class 11
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
        client_idcs = dirichlet_split_noniid(trainy[train_data_known_idx],total_class, alpha=dirichlet_alpha, n_clients=client_num, state=state)

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



    if args.setting == 2 :
        # Load CIFAR-10 as known class dataset with known class range 0-5 and unknown class as (5-10)->mapped -> 5
        total_class = args.known_class  # CIFAR-10 classes
        unknown_label = args.known_class  # Assign CIFAR-10 (5-10) data to class 5
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        # Known index
        indx_0_to_4 = [i for i, label in enumerate(train_dataset.targets) if label in range(args.known_class)]  # Train
        indices_0_to_4 = [i for i, label in enumerate(test_dataset.targets) if label in range(args.known_class)]  # Test
        ## Unknown ndex ## 
        indx_5_to_9 = [i for i, label in enumerate(train_dataset.targets) if label in range(args.known_class, 10)]  
        indices_5_to_9 = [i for i, label in enumerate(test_dataset.targets) if label in range(args.known_class, 10)]

        ##### Creating known and unknown testset ####
        trainset = Subset(train_dataset,indx_0_to_4)
        testset = Subset(test_dataset,indices_0_to_4) 
        trainset_unknown = Subset(train_dataset, indx_5_to_9)
        testset_unknown = Subset(test_dataset, indices_5_to_9)
        testset_unknown = CustomCIFAR10Wrapper_test(testset_unknown, num_class=5)
        trainset_unknown = CustomCIFAR10Wrapper_test(trainset_unknown, num_class=5)
        merged_unknown_dataset = ConcatDataset([testset_unknown, trainset_unknown])
        trainx = np.array(train_dataset.data)[indx_0_to_4]
        # trainy = np.array(train_dataset.targets)[indx_0_to_4]
        # train_data_known_idx = indx_0_to_4
        np.random.shuffle(indx_0_to_4)
        val_size = int(len(indx_0_to_4) * val_split)
        val_data_known_idx, train_data_known_idx = indx_0_to_4[:val_size], indx_0_to_4[val_size:]
        trainy = np.array(train_dataset.targets)[train_data_known_idx]
        # print(val_data_known_idx)

        num_workers = 0 if platform.system() == 'Windows' else 4
        client_idcs = dirichlet_split_noniid(trainy, total_class,alpha=dirichlet_alpha, n_clients=client_num, state=state)

        trainloaders = []
        # out = 0
        # for i in range(len(client_idcs)):
        #     out += len(client_idcs[i])
        # print(out,trainy.shape ) #ok
        for i in range(client_num):
            sub_train_idx = client_idcs[i] #np.array(train_data_known_idx)[np.array(client_idcs[i], dtype=int)]
            # print(sub_train_idx)
            client_trainset = data.Subset(trainset, sub_train_idx)
            client_loader = data.DataLoader(client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
            trainloaders.append(client_loader)

        valloader = data.DataLoader(data.Subset(train_dataset, val_data_known_idx), batch_size=batchsize, shuffle=False, num_workers=num_workers)

        # merged_test_dataset = CustomDataset(testx, testy, transform=transform)
        closeloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)
        openloader = data.DataLoader(merged_unknown_dataset , batch_size=1, shuffle=False, num_workers=num_workers)


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
    
class CustomCIFAR10Wrapper_test(Dataset):
    """
    A wrapper for datasets to assign a fixed label to all samples.
    """
    def __init__(self, original_dataset, num_class):
        self.original_dataset = original_dataset
        self.num_class = num_class

    def __getitem__(self, index):
        img, _ = self.original_dataset[index]  # Get the image and ignore the original label
        target = self.num_class  # Map all labels to a single value
        return img, target   # Return the image with the fixed unknown label

    def __len__(self):
        return len(self.original_dataset)


def dirichlet_split_noniid(train_labels, num_class, alpha, n_clients, state):
    np.random.set_state(state)  # Ensure reproducibility
    total_classes = num_class # Number of unique classes

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
        while len(client_classes[i]) < total_classes:
            # Get missing classes
            missing_classes = set(range(total_classes)) - client_classes[i]
            if not missing_classes:
                break 
            new_class = np.random.choice(list(missing_classes))  # Pick random missing class

            # Add random 5 samples from the missing class
            extra_samples = np.random.choice(class_indices[new_class], size=5, replace=False)
            client_indices[i].extend(extra_samples)
            client_classes[i].add(new_class)

        while len(client_classes[i]) > total_classes:
            # Remove excess class
            excess_class = np.random.choice(list(client_classes[i]))
            client_indices[i] = [idx for idx in client_indices[i] if train_labels[idx] != excess_class]
            client_classes[i].remove(excess_class)

        client_indices[i] = np.array(client_indices[i])

    return client_indices





