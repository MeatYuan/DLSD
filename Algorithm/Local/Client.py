from Algorithm.base import BaseClient
import torch
import torch.nn as nn
from utils.train import Train_model, Evaluate
class Client(BaseClient):
    def __init__(self, model, local_train_loader, local_test_loader, client_id,
                 args, device='cpu'):
        super().__init__(model, local_train_loader, local_test_loader,
                         client_id, args, device)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
        self.print_info()


    def train_local_model(self, args):
        loss, acc = Train_model(self.local_model, 
                                self.train_loader, 
                                optimizer = self.optimizer,
                                epochs = args.local_epochs, 
                                device = self.device)
        return loss, acc
     

    

