import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import numpy as np

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class FedCifar10():
    def __init__(self,args=None):
        self.num_clients = 10 # args.num_client
        self.batch_size = 32 #args.batch_size
        self.alpha = 0.5 #args.dirichlet
        self.known_class = 7 # args.known_class
        self.unknown_class = 3 #args.unknown_class
        
        total_class = 10
        assert self.known_class + self.unknown_class <= total_class
        self.known_class_list, self.unknown_class_list = self.split_known_unknown_classes(total_class)
        print(f"Known classes: {self.known_class_list}")
        print(f"Unknown classes: {self.unknown_class_list}")
    
    def split_known_unknown_classes(self, total_class):
        class_list = np.arange(total_class)
        np.random.seed(42)
        np.random.shuffle(class_list)
        known_class_list = class_list[:self.known_class]
        unknown_class_list = class_list[self.known_class:]
        return known_class_list, unknown_class_list
    

    def relabel_and_filter(self, dataset, is_train=True):
        """
        Relabels known and unknown classes, and filters unknown classes from training and validation sets.
        """
        known_dict = {old_label: new_label for new_label, old_label in enumerate(self.known_class_list)}
        unknown_dict = {old_label: new_label + len(self.known_class_list) for new_label, old_label in enumerate(self.unknown_class_list)}

        def relabel(batch):
            labels = np.array(batch['label'])
            new_labels = np.copy(labels)
            
            # Relabel known classes
            for old_label, new_label in known_dict.items():
                new_labels[labels == old_label] = new_label
                
            # Relabel unknown classes
            for old_label, new_label in unknown_dict.items():
                new_labels[labels == old_label] = new_label
            
            # Filter unknowns if training or validation
            if is_train:
                known_mask = np.isin(labels, self.known_class_list)
                batch['img'] = [img for i, img in enumerate(batch['img']) if known_mask[i]]
                batch['label'] = new_labels[known_mask]
            else:
                batch['label'] = new_labels
                
            return batch
        
        return dataset.with_transform(relabel)

    def load_datasets(self,partition_id: int):

        # Partitioner for federated dataset
        partitioner = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                   alpha=self.alpha, min_partition_size=10,
                                   self_balancing=True)
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner })
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            # Instead of passing transforms to CIFAR10(..., transform=transform)
            # we will use this function to dataset.with_transform(apply_transforms)
            # The transforms object is exactly the same
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        # Relabel and filter datasets
        
        partition_train_test = self.relabel_and_filter(partition_train_test, is_train=True)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=self.batch_size, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=self.batch_size)
        
        # Test set includes all classes
        testset = fds.load_split("test").with_transform(apply_transforms)
        testset = self.relabel_and_filter(testset, is_train=False)
        testloader = DataLoader(testset, batch_size=self.batch_size)
        return trainloader, valloader, testloader
    

if __name__=='__main__':

    import matplotlib.pyplot as plt
    fedcifar = FedCifar10()
    trainloader, _, _ = fedcifar.load_datasets(partition_id=0)
    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()

    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()