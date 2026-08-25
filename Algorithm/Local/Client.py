import torch
import torch.nn as nn
from utils.train import Train_model, Evaluate
class Client:
    def __init__(self, model, local_train_loader, local_test_loader, client_id, local_val_loader = None, device = "gpu", args = None):
        # client id
        self.client_id = client_id

        # local model
        self.local_model = model
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        # data loader
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader

        self.device = device
        self.print_info()

    def get_id(self):
        """Return the client ID."""
        return self.client_id

    def print_info(self):
        """Print the client information."""
        print("============================================")
        print(f"Client ID: {self.client_id}")
        print(f"Local Model: {self.local_model.__class__.__name__}")
        print(f"Train Loader: {len(self.train_loader.dataset)}")
        print(f"Test Loader: {len(self.test_loader.dataset)}")
        #print(f"Validation Loader: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("============================================")


    def train_local_model(self, args):
        loss, acc = Train_model(self.local_model, 
                                self.train_loader, 
                                optimizer = self.optimizer,
                                epochs = args.local_epochs, 
                                device = self.device)
        return loss, acc
     

    def evaluate(self,):
        loss, acc = Evaluate(self.local_model, self.test_loader, device = self.device)
        return loss, acc
    

