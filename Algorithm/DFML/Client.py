import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils.train import Evaluate, WSM_loss, KL_divergence, offload_optimizer_states, onload_optimizer_states


class Client:
    def __init__(self, model, local_train_loader, local_test_loader, client_id,
                 local_val_loader=None, device="gpu", args=None):
        # client id
        self.client_id = client_id

        # local (regular) model Wn
        self.local_model = model

        # data loader
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader

        self.device = device
        self.args = args

        # DFML state
        self.peak_model = deepcopy(model)        # Wn_hat: frozen copy holding best global params
        self.local_alpha = 0.0                   # alpha_n: largest alpha seen so far
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Phi_n
        self.class_dist = self._compute_class_dist(args.num_classes)  # beta over local data

        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.print_info()

    def _compute_class_dist(self, num_classes):
        """Per-class proportion (beta) of this client's local data, used by WSM."""
        counts = torch.zeros(num_classes)
        for _, labels in self.train_loader:
            counts += torch.bincount(labels, minlength=num_classes).float()
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts

    def get_id(self):
        """Return the client ID."""
        return self.client_id

    def print_info(self):
        """Print the client information."""
        print("============================================")
        print(f"Client ID: {self.client_id}")
        print(f"Local Model: {self.local_model.__class__.__name__}")
        print(f"Model Size (Phi): {self.num_params}")
        print(f"Train Loader: {len(self.train_loader.dataset)}")
        print(f"Test Loader: {len(self.test_loader.dataset)}")
        print(f"Device: {self.device}")
        print("============================================")

    # --- accessors used by the aggregator during mutual learning ---
    def get_model(self):
        return self.local_model

    def get_size(self):
        return self.num_params

    def get_optimizer(self):
        return self.optimizer

    def local_train(self):
        """Local training on the client's own data (DFML client side, Eq 2)."""
        model = self.local_model.to(self.device)
        onload_optimizer_states(self.device, self.optimizer)
        model.train()

        use_wsm = self.args.dfml_use_wsm
        avg_loss, avg_acc = 0.0, 0.0
        for epoch in range(self.args.local_epochs):
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)['logits']
                if use_wsm:
                    loss = WSM_loss(outputs, labels, self.class_dist)
                else:
                    loss = F.cross_entropy(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)
            avg_loss += total_loss / total_samples
            avg_acc += total_correct / total_samples

        avg_loss /= self.args.local_epochs
        avg_acc /= self.args.local_epochs

        offload_optimizer_states(self.optimizer)
        model.to('cpu')
        model.eval()
        torch.cuda.empty_cache()
        return avg_loss, avg_acc

    def mutual_learning(self, participants, alpha, args):
        """Aggregator side: weighted mutual learning over THIS client's data
           (DFML Eq 1 & 3). `self` is the aggregator; `participants` is the list
           of participating Client objects (senders + aggregator). All participant
           models are updated in place: each is the student while the others act as
           detached teachers, weighted by model size Phi."""
        if len(participants) < 2:
            return  # no teachers available, nothing to distill

        device = self.device
        beta_a = self.class_dist  # WSM uses the aggregator's class distribution

        models = [c.get_model().to(device) for c in participants]
        optims = [c.get_optimizer() for c in participants]
        phi = [c.get_size() for c in participants]

        for m in models:
            m.train()
        for opt in optims:
            onload_optimizer_states(device, opt)

        T = args.dfml_T
        use_wsm = args.dfml_use_wsm
        n_part = len(models)

        # Teacher weights, fixed for the whole round: kl_w[n][q] is how much teacher q
        # counts towards student n. 'phi' is Eq 3 (weighted by trainable parameters);
        # 'uniform' is the vanilla average of Appendix A.4.7. Each row sums to 1.
        weighting = getattr(args, 'dfml_kl_weighting', 'phi')
        kl_w = []
        for n in range(n_part):
            if weighting == 'uniform':
                row = [0.0 if q == n else 1.0 / (n_part - 1) for q in range(n_part)]
            else:
                denom = sum(phi[q] for q in range(n_part) if q != n)
                row = [0.0 if (q == n or denom == 0) else phi[q] / denom for q in range(n_part)]
            kl_w.append(row)

        for k in range(args.dfml_K):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Snapshot teacher logits before any update this batch (Algorithm 1).
                with torch.no_grad():
                    snaps = [m(inputs)['logits'].detach() for m in models]

                for n in range(n_part):
                    outputs = models[n](inputs)['logits']
                    if use_wsm:
                        sup = WSM_loss(outputs, labels, beta_a)
                    else:
                        sup = F.cross_entropy(outputs, labels)

                    kl = 0.0
                    for q in range(n_part):
                        if q == n or kl_w[n][q] == 0.0:
                            continue
                        kl = kl + kl_w[n][q] * KL_divergence(snaps[q], outputs, T=T)

                    loss = (1 - alpha) * sup + alpha * kl

                    optims[n].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(models[n].parameters(), max_norm=5.0)
                    optims[n].step()

        for opt in optims:
            offload_optimizer_states(opt)
        for m in models:
            m.to('cpu')
            m.eval()
        torch.cuda.empty_cache()

    def update_peak(self, alpha, in_peak_window=False):
        """Update the peak model when the current alpha is at least the largest alpha
           previously used to aggregate this client (Algorithm 1), or when this round
           falls inside the cycle's top-n alpha window (Appendix A.4.6, which applies
           when participation is below 50%).

           alpha_n only ever moves up: a window update at a sub-maximal alpha must not
           lower the gate, or every later round would pass `alpha >= alpha_n` and the
           peak model would degrade into a copy of the regular model."""
        if alpha >= self.local_alpha or in_peak_window:
            self.peak_model.load_state_dict(deepcopy(self.local_model.state_dict()))
            self.local_alpha = max(self.local_alpha, alpha)
    
    def evaluate(self):
        """Global accuracy, read from the peak models unless dfml_eval_peak is off
           (Section 4.1: evaluated by examining the peak models unless stated
           otherwise)."""
        model = self.peak_model if self.args.dfml_eval_peak else self.local_model
        loss, acc = Evaluate(model, self.test_loader, device=self.device)
        return loss, acc
