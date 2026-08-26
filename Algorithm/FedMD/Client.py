from Algorithm.base import BaseClient
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.train import Evaluate


class Client(BaseClient):
    """
    FedMD
    Reference: D. Li and J. Wang, "FedMD: Heterogenous Federated Learning via Model Distillation,", NeurIPS 2019.
    Modified with reference to: https://github.com/diogenes0319/FedMD_clean
    """

    def __init__(self, model, local_train_loader, local_test_loader,
                 client_id, args, device='cpu'):
        super().__init__(model, local_train_loader, local_test_loader,
                         client_id, args, device)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        self.revisit_epochs = args.revisit_epochs
        self.digest_epochs = args.digest_epochs

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_T = args.fedmd_T

        self.print_info()


    # ── pre-training on public data ──────────────────────────────────────────

    def public_pretrain(self, epochs, public_loader):
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()
        for _ in range(epochs):
            for X, y in public_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.local_model(X)['logits']
                loss = self.ce_loss(logits, y)
                loss.backward()
                self.optimizer.step()
        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

    # ── pre-training on private data ─────────────────────────────────────────

    def pretrain(self, epochs):
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()
        for _ in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.local_model(X)['logits']
                loss = self.ce_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                self.optimizer.step()
        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

    # ── communication step: upload logits ───────────────────────────────────

    def compute_logits(self, public_loader):
        self.local_model = self.local_model.to(self.device)
        self.local_model.eval()
        all_logits = []
        with torch.no_grad():
            for X, in public_loader:
                X = X.to(self.device)
                logits = self.local_model(X)['logits']
                all_logits.append(logits.cpu())
        self.local_model.to('cpu')
        torch.cuda.empty_cache()
        return torch.cat(all_logits, dim=0)  # (N, C)

    # ── digest step: KD alignment with mean logits ───────────────────────────

    def digest(self, public_loader, mean_logits):
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()

        for _ in range(self.digest_epochs):
            offset = 0
            for X, in public_loader:
                X = X.to(self.device)
                bsz = X.size(0)
                teacher = mean_logits[offset:offset + bsz].to(self.device).detach()
                offset += bsz

                self.optimizer.zero_grad()
                logits = self.local_model(X)['logits']
                T = self.kd_T
                loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(teacher / T, dim=-1),
                    reduction='batchmean',
                ) * (T ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10.0)
                self.optimizer.step()

        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

    # ── local training on private data (revisit) ────────────────────────────

    def train(self):
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()
        for _ in range(self.revisit_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.local_model(X)['logits']
                loss = self.ce_loss(logits, y)
                loss.backward()
                self.optimizer.step()
        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

