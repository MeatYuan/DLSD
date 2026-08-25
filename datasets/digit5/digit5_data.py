import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvd
from torchvision import transforms

from datasets.Data import Dataset_Mean_Std
from datasets.digit5.synthetic_digits import SyntheticDigits
from datasets.digit5.mnist_m import MNISTM


DOMAINS = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]
_GRAY_DOMAINS = {"MNIST", "USPS"}
MEAN, STD = Dataset_Mean_Std("Digit5")

def _build_transform(image_size, is_gray):
    ops = []
    if is_gray:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops += [
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),  # -> [-1,1]; denorm x*0.5+0.5 -> [0,1]
    ]
    return transforms.Compose(ops)


def _load_domain(name, root, train, transform):
    """Return a torchvision-style dataset for one domain / split."""
    if name == "MNIST":
        return tvd.MNIST(root=os.path.join(root, "MNIST"), train=train, download=True, transform=transform)
    if name == "USPS":
        return tvd.USPS(root=os.path.join(root, "USPS"), train=train, download=True, transform=transform)
    if name == "SVHN":
        return tvd.SVHN(root=os.path.join(root, "SVHN"), split="train" if train else "test",
                        download=True, transform=transform)
    if name == "SynthDigits":
        return SyntheticDigits(root=root, train=train, download=True, transform=transform)
    if name == "MNIST_M":
        return MNISTM(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unknown domain: {name}")


def _get_targets(ds):
    """Extract integer label array from any of the dataset types used here."""
    if hasattr(ds, "targets"):
        t = ds.targets
    elif hasattr(ds, "labels"):          # SVHN
        t = ds.labels
    else:
        raise AttributeError(f"{type(ds).__name__} has no targets/labels")
    if torch.is_tensor(t):
        t = t.cpu().numpy()
    return np.asarray(t).astype(int)


def _balanced_shards(targets, n_shards, num_classes, seed, percent=1.0, samples_per_domain=None):
    """
    Split indices into `n_shards` label-balanced IID shards.
    Each shard gets ~equal count per class -> 'label 域內均衡'.

    `percent`            : optionally keep only this fraction of each class (proportional).
    `samples_per_domain` : optionally CAP the domain to this many samples total, split
                           evenly across classes (~samples_per_domain/num_classes each).
                           Use this to EQUALISE domain sizes so data quantity is not a
                           confound with domain shift (e.g. cap every domain to ~7000).
    """
    rng = np.random.RandomState(seed)
    cap_per_class = None
    if samples_per_domain is not None:
        cap_per_class = max(1, samples_per_domain // num_classes)

    shards = [[] for _ in range(n_shards)]
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        if percent < 1.0:
            idx_c = idx_c[: max(1, int(len(idx_c) * percent))]
        if cap_per_class is not None:
            idx_c = idx_c[:cap_per_class]
        for j, chunk in enumerate(np.array_split(idx_c, n_shards)):
            shards[j].extend(chunk.tolist())
    for s in shards:
        rng.shuffle(s)
    return shards


def Create_Digit5_Data(args, num_workers=0, root="./Data"):
    assert args.num_clients % len(DOMAINS) == 0, \
        f"num_clients ({args.num_clients}) must be divisible by {len(DOMAINS)} for even domain split"
    clients_per_domain = args.num_clients // len(DOMAINS)
    percent = float(getattr(args, "percent", 1.0))
    # cap every domain to the same total size so data quantity is not a confound.
    samples_per_domain = getattr(args, "samples_per_domain", None)

    client_train_loaders, client_test_loaders, client_domain = [], [], []
    domain_test_loaders = {}

    for d, name in enumerate(DOMAINS):
        is_gray = name in _GRAY_DOMAINS
        tf = _build_transform(args.image_size, is_gray)

        train_datasets = _load_domain(name, root, train=True, transform=tf)
        test_datasets = _load_domain(name, root, train=False, transform=tf)

        # per-domain test loader (shared by that domain's clients = intra-domain eval)
        test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        domain_test_loaders[name] = test_loader

        # label-balanced split of this domain's train set across its clients
        targets = _get_targets(train_datasets)
        shards = _balanced_shards(targets, clients_per_domain, args.num_classes,
                                  seed=args.random_seed + d, percent=percent,
                                  samples_per_domain=samples_per_domain)

        for j in range(clients_per_domain):
            sub = Subset(train_datasets, shards[j])
            loader = DataLoader(sub, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
            client_train_loaders.append(loader)
            client_test_loaders.append(test_loader)
            client_domain.append(name)

    return {
        "client_train": client_train_loaders,
        "client_test": client_test_loaders,
        "domain_test": domain_test_loaders,
        "client_domain": client_domain,
    }

