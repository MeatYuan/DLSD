import torch
import numpy as np
from torch.utils.data import Sampler, DataLoader, TensorDataset, Subset
from collections import defaultdict

class ClassBatchSampler(Sampler):
    def __init__(self, dataset, client_id, batch_size = 50, shuffle=False, drop_last=True):
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = dataset.indices
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))
        
        if hasattr(base_dataset, "labels"):          # e.g., SVHN
            all_targets = base_dataset.labels
        elif hasattr(base_dataset, "targets"):       # e.g., CIFAR10, MNIST
            all_targets = base_dataset.targets
        else:
            raise AttributeError("Dataset does not have 'labels' or 'targets' attribute.")
        
        targets = [all_targets[i] for i in indices]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.class_indices = defaultdict(list)
        for idx, target in enumerate(targets):
            if isinstance(target, torch.Tensor):
                target_item = target.item()
            else:
                target_item = target
            self.class_indices[target_item].append(idx)

        self.batches = []
        
        classes = list(self.class_indices.keys())
        if self.shuffle:
            np.random.shuffle(classes)

        label_dict = {}
        for class_label in classes:
            class_idx_list = self.class_indices[class_label]
            if self.shuffle:
                np.random.shuffle(class_idx_list) 

            # 該class的資料總數
            n_samples = len(class_idx_list)
            
            # 可以分成幾組batch
            n_batches = n_samples // self.batch_size
            
            label_dict[class_label] = n_batches

            # 包裝成batch
            for i in range(n_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch = class_idx_list[start:end]
                self.batches.append(batch)
            
            # 看是否也要把剩餘資料包裝進去
            if not self.drop_last:
                remaining = n_samples % self.batch_size
                if remaining > 0:
                    batch = class_idx_list[n_batches*self.batch_size:]
                    self.batches.append(batch)
                label_dict[class_label] = n_batches + 1
        print(f"Client {client_id} Each class batch counts: {label_dict}")
    
    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, logits):
        self.features = features
        self.labels = labels
        self.logits = logits

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.logits[idx]

class SynDataset(torch.utils.data.Dataset):
    def __init__(self, images, features=None, logits=None):
        self.images = images
        self.features = features
        self.logits = logits

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.features is not None and self.logits is not None:
            feature = self.features[idx]
            logit = self.logits[idx]
            return img, feature, logit
        
        return img
    
    def update_features_logits(self, features, logits):
        self.features = features
        self.logits = logits