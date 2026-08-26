import numpy as np
import random
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset, Subset
import math
from collections import Counter
def Dataset_Mean_Std(dataset):
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    elif dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif dataset == 'Digit5':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    return mean, std
    
def Get_Data(args):
    
    if args.dataset == 'MNIST':
        mean, std = Dataset_Mean_Std(args.dataset)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size),  # Resize to match CIFAR10 input size
            transforms.Normalize(mean, std) 
        ])
        train_dataset = datasets.MNIST(root='./Data/MNIST', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./Data/MNIST', train=False, download=True, transform=transform)
        val_dataset = datasets.MNIST(root='./Data/MNIST', train=True, download=True, transform=transform)
        num_classes = args.num_classes
    elif args.dataset == 'CIFAR10':
        mean, std = Dataset_Mean_Std(args.dataset)

        train_tf = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR10('./Data/CIFAR10', train=True, download=True, transform=train_tf)
        test_dataset = datasets.CIFAR10('./Data/CIFAR10', train=False, download=True, transform=test_tf)
        val_dataset = datasets.CIFAR10('./Data/CIFAR10', train=True, download=True, transform=test_tf)
        num_classes = args.num_classes
    elif args.dataset == 'SVHN':
        mean, std = Dataset_Mean_Std(args.dataset)

        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.SVHN('./Data/SVHN', split='train', download=True, transform=train_tf)
        test_dataset = datasets.SVHN('./Data/SVHN', split='test', download=True, transform=test_tf)
        val_dataset = datasets.SVHN('./Data/SVHN', split='train', download=True, transform=test_tf)
        num_classes = args.num_classes
    elif args.dataset == 'CIFAR100':
        mean, std = Dataset_Mean_Std(args.dataset)

        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR100('./Data/CIFAR100', train=True, download=True, transform=train_tf)
        test_dataset = datasets.CIFAR100('./Data/CIFAR100', train=False, download=True, transform=test_tf)
        val_dataset = datasets.CIFAR100('./Data/CIFAR100', train=True, download=True, transform=test_tf)
        num_classes = args.num_classes
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))
    

    return train_dataset, test_dataset, val_dataset, num_classes

#Origin resource : https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/datasets/partition_data.py
def build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    #use workers to support the partitioning.
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch
        

def Create_Data(args):
    """
    Split dataset into num_clients clients using Dirichlet distribution(noniid).
    """
    # ── Digit-5 domain-shift setup (isolated builder under datasets/digit5/) ──
    # Each client belongs to one visual domain (MNIST/SVHN/USPS/SynthDigits/MNIST-M).
    # client_test is the union of all 5 domains (for the generic per-round eval);
    # domain_test + client_domain are carried through so eval_digit5 can compute the
    # intra/inter-domain accuracy matrix.
    if args.dataset == 'Digit5':
        from torch.utils.data import ConcatDataset
        from datasets.digit5.digit5_data import Create_Digit5_Data
        d5 = Create_Digit5_Data(args, num_workers=2)
        global_test_ds = ConcatDataset([ld.dataset for ld in d5["domain_test"].values()])
        global_test = DataLoader(global_test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2)
        for i, dom in enumerate(d5["client_domain"]):
            print(f"Client {i} domain: {dom}, train size: {len(d5['client_train'][i].dataset)}")
        return {"client_train": d5["client_train"], "client_test": global_test,
                "domain_test": d5["domain_test"], "client_domain": d5["client_domain"]}

    random_state = np.random.RandomState(args.Noniid_random_state_seed)
    
    # 獲取Dataset
    train_dataset, test_dataset, val_dataset, num_classes = Get_Data(args)

    # 取得labels 不經過transforms
    if hasattr(train_dataset, "labels"):          # e.g., SVHN
        train_labels = train_dataset.labels
    elif hasattr(train_dataset, "targets"):       # e.g., CIFAR10, MNIST
        train_labels = train_dataset.targets
    else:
        raise AttributeError("Dataset does not have 'labels' or 'targets' attribute.")

    # 得到每一個client Dataset 的 index
    idx_batch = build_non_iid_by_dirichlet(
        random_state=random_state,
        indices2targets=list(enumerate(train_labels)),
        non_iid_alpha=args.noniid_alpha,
        num_classes=num_classes,
        num_indices=len(train_dataset),
        n_workers=args.num_clients,
    )

    # Client train loaders
    client_train_loaders = []
    for client_id in range(args.num_clients):
        # # 產生 Train DataLoader
        train_subset = Subset(train_dataset, idx_batch[client_id])
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        # 存放到list
        client_train_loaders.append(train_loader)

    labels_distribution = []
    for i, indices in enumerate(idx_batch):
        labels = [int(train_labels[idx]) for idx in indices]
        label_count = Counter(labels)
        labels_distribution.append(label_count)
        print(f"Client {i} label distribution: {dict(label_count)}")

    client_test_loaders = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    re = {"client_train": client_train_loaders, "client_test": client_test_loaders}

    return re


def Get_FedMD_Public_Dataset(args):
    """
    Randomly sample 10 CIFAR-100 classes, excluding classes 50-59
    (which are used as CIFAR-10 proxies in this codebase), then relabel to 0-9.
    Selection is reproducible via args.Noniid_random_state_seed.
    """
    CIFAR10_PROXY = set(range(50, 60))
    candidates = [c for c in range(100) if c not in CIFAR10_PROXY]
    rng = np.random.RandomState(args.Noniid_random_state_seed)
    public_classes = sorted(rng.choice(candidates, 10, replace=False).tolist())
    print(f"[FedMD] Public CIFAR-100 classes: {public_classes}")

    mean, std = Dataset_Mean_Std('CIFAR10')
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_ds = datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=True, transform=transform)
    targets = np.array(full_ds.targets)

    mask = np.isin(targets, public_classes)
    indices = np.where(mask)[0].tolist()
    relabel = {cls: i for i, cls in enumerate(public_classes)}
    relabeled_targets = [relabel[int(targets[i])] for i in indices]

    class RelabeledSubset(torch.utils.data.Dataset):
        def __init__(self, subset, labels):
            self.subset = subset
            self.labels = labels
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, _ = self.subset[idx]
            return x, self.labels[idx]

    return RelabeledSubset(Subset(full_ds, indices), relabeled_targets)


def Get_FedMD_Alignment_Dataset(args):
    """
    Full CIFAR-100 (all 100 classes, 50,000 images).
    Used for per-round alignment sampling in FedMD (labels ignored during digest).
    """
    mean, std = Dataset_Mean_Std('CIFAR100')
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=True, transform=transform)
