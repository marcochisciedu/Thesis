import torch
import torchvision as tv

from typing import List


# Takes a list of indices (CIFAR100 classes) and creates a subset
class CIFAR100Subset(tv.datasets.CIFAR100):
    def __init__(self, subset: List[int], **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        # Check indices
        assert max(subset) <= max(self.targets)
        assert min(subset) >= min(self.targets)
        # Selects the classes
        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if label in subset:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in self.subset]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return super().__getitem__(self.aligned_indices[item])

# Creates and returns train and validation dataloaders
def create_dataloaders(dataset, path, batch_size, input_size=32, subset_list = None,
                 num_workers = 0):
    if dataset == "cifar10":
        transform = tv.transforms.Compose([tv.transforms.Resize((input_size, input_size)),
                                    tv.transforms.ToTensor(),
                                    tv.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                        (0.2675, 0.2565, 0.2761))
                                    ])
        # Can be used as a gallery set
        valid_set = tv.datasets.CIFAR10(root=path, 
                                    train=False, 
                                    download=True, 
                                    transform=transform
                                )
        
        # Can be used as a query set
        train_set = tv.datasets.CIFAR10(root=path, 
                                train=True, 
                                download=True, 
                                transform=transform
                                )    
    elif dataset == "cifar100":
        train_transform = tv.transforms.Compose(
                        [tv.transforms.Resize((input_size, input_size)),
                        tv.transforms.RandomCrop(input_size, padding=4),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                        ])
        val_transform =tv.transforms.Compose(
                    [tv.transforms.Resize((input_size, input_size)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))])
        if subset_list == None:
            train_set = tv.datasets.CIFAR100(root = path, transform=train_transform, train=True, download=True)
            valid_set = tv.datasets.CIFAR100(root = path, transform=val_transform, train=False, download=True)
        else:         # take a Subset of CIFAR100
            train_set = CIFAR100Subset(subset=subset_list ,root=path, train=True, download=True, transform=train_transform)
            print(f'Cifar100 subset classes: {train_set.get_class_names()}')
            valid_set = CIFAR100Subset(subset=subset_list ,root=path, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                        f"{dataset} dataset in the PyTorch codebase. "
                        f"In principle, it should be easy to add :)")

    print(f"Using a training set with {len(train_set)} images.")
    print(f"Using a validation set with {len(valid_set)} images.")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True, drop_last=False)
    
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers, pin_memory=True, drop_last=False)

    return  train_loader, valid_loader