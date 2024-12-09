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

# Returns train and validation datasets
def load_dataset(dataset, path, batch_size, input_size=32, subset_list = None,
                  batch_split = 1, num_workers = 0):

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
        train_transform = [tv.transforms.Resize((input_size, input_size)),
                        tv.transforms.RandomCrop(input_size, padding=4),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                        ]
        val_transform = [tv.transforms.Resize((input_size, input_size)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))
                    ]
        if subset_list == None:
            train_set = tv.datasets.CIFAR100(data_path = path, transform=train_transform, train=True, download=True)
            valid_set = tv.datasets.CIFAR100(data_path = path, transform=val_transform, train=False, download=True)
        else:
            train_set = CIFAR100Subset(subset=subset_list ,root=path, train=True, download=True, transform=train_transform)
            valid_set = CIFAR100Subset(subset=subset_list ,root=path, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                        f"{dataset} dataset in the PyTorch codebase. "
                        f"In principle, it should be easy to add :)")

    print(f"Using a training set with {len(train_set)} images.")
    print(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = batch_size// batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader