import torch


class ProxyDataset(torch.utils.data.Dataset):
    """The synthetic images a client shares with its neighbours, paired with the
    features and logits its own model produced for them.

    Features and logits are optional so the images can be held before the owning
    model has been run over them; once both are present each item is the triple
    the receiving client distils from.
    """

    def __init__(self, images, features=None, logits=None):
        self.images = images
        self.features = features
        self.logits = logits

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.features is not None and self.logits is not None:
            return self.images[idx], self.features[idx], self.logits[idx]
        return self.images[idx]
