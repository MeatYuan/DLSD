from Algorithm.base import BaseClient
import torch
import torch.nn as nn
from utils.train import Evaluate
from torch.nn import CrossEntropyLoss


class Client(BaseClient):
    """
    FedProto client.
    Reference: Tan et al., AAAI 2022. github.com/yuetan031/FedProto

    Prototype = classifier[:-1](feature) in shared latent space (proto_dim).
    Latent layer added in main.py so all heterogeneous models share the same
    prototype space. Per-label aggregation as in original paper.
    """

    def __init__(self, model, local_train_loader, local_test_loader,
                 client_id, args, device='cpu'):
        super().__init__(model, local_train_loader, local_test_loader,
                         client_id, args, device)
        self.train_ep = args.train_ep
        self.ld = args.ld
        self.train_samples = len(local_train_loader.dataset)

        self.criterion = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr, momentum=args.momentum)

        print(f"[FedProto] Client {client_id} | model={model.__class__.__name__} "
              f"| train={self.train_samples}")


    def train(self, global_protos):
        """
        Loss = CE(logits, y) + ld * MSE(latent, global_proto[y])
        protos = classifier[:-1](feature) — shared latent space.
        Returns agg_protos_label: {label: [proto tensors]} for server aggregation.
        """
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()

        agg_protos_label = {}

        for _ in range(self.train_ep):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.local_model.zero_grad()
                out = self.local_model(images)
                logits = out['logits']
                protos = self.local_model.classifier[:-1](out['feature'])  # (B, proto_dim)

                loss1 = self.criterion(logits, labels)

                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = protos.data.clone()
                    for i, label in enumerate(labels):
                        if label.item() in global_protos:
                            proto_new[i] = global_protos[label.item()][0].to(self.device)
                    loss2 = self.loss_mse(proto_new, protos)

                loss = loss1 + loss2 * self.ld
                loss.backward()
                self.optimizer.step()

                for i, label in enumerate(labels):
                    lbl = label.item()
                    if lbl in agg_protos_label:
                        agg_protos_label[lbl].append(protos[i].detach().cpu())
                    else:
                        agg_protos_label[lbl] = [protos[i].detach().cpu()]

        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

        return agg_protos_label

    def evaluate(self, global_protos=None):
        """
        If global_protos provided: nearest-neighbor in latent space (Eq. 4 in paper).
        ŷ = argmin_j ||classifier[:-1](feature) − global_proto[j]||₂
        Otherwise: fall back to classifier logits.
        """
        if not global_protos:
            loss, acc = Evaluate(self.local_model, self.test_loader, device=self.device)
            return loss, acc

        self.local_model = self.local_model.to(self.device)
        self.local_model.eval()

        proto_classes = sorted(global_protos.keys())
        proto_matrix = torch.stack(
            [global_protos[c][0] for c in proto_classes]
        ).to(self.device)  # (num_classes, proto_dim)

        ce = CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.local_model(images)
                embeddings = self.local_model.classifier[:-1](out['feature'])  # (B, proto_dim)

                dists = torch.cdist(embeddings, proto_matrix)  # (B, num_classes)
                pred_idx = dists.argmin(dim=1)
                preds = torch.tensor(
                    [proto_classes[i] for i in pred_idx.cpu().tolist()],
                    device=self.device
                )

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += ce(out['logits'], labels).item() * labels.size(0)

        self.local_model.to('cpu')
        torch.cuda.empty_cache()
        return total_loss / total_samples, total_correct / total_samples
