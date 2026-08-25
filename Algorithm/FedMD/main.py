import torch
from torch.utils.data import DataLoader, TensorDataset

from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.FedMD.Client import Client
from Algorithm.FedMD.default_args import apply_fedmd_defaults
from datasets.Data import Get_FedMD_Alignment_Dataset, Get_FedMD_Public_Dataset
from utils.network import Sample_Client


def FedMD(Client_Data, models, model_list, args):
    """
    FedMD
    Reference: D. Li and J. Wang, “FedMD: Heterogenous Federated Learning via Model Distillation,”, NeurIPS 2019.
    Modified with reference to: https://github.com/diogenes0319/FedMD_clean
    """
    #read default args
    apply_fedmd_defaults(args)
    recorder = MetricsRecorder(args)

    # alignment sampler: random 5000 from full CIFAR-100 each round
    alignment_ds = Get_FedMD_Alignment_Dataset(args)
    alignment_sampler = DataLoader(
        alignment_ds, batch_size=args.public_size, shuffle=True, num_workers=2)
    alignment_iter = iter(alignment_sampler)

    client_list = [
        Client(
            model=models[i],
            local_train_loader=Client_Data['client_train'][i],
            local_test_loader=Client_Data['client_test'],
            client_id=i,
            args=args,
            device=args.device,
        )
        for i in range(args.num_clients)
    ]

    # ── public pretrain ───────────────────────────────────────────────────────
    if args.public_pretrain_epochs > 0:
        print(f"\n===== FedMD Public pretrain (epochs={args.public_pretrain_epochs}) =====")
        public_ds = Get_FedMD_Public_Dataset(args)
        public_loader = DataLoader(public_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        for client in client_list:
            client.public_pretrain(epochs=args.public_pretrain_epochs, public_loader=public_loader)

    # ── private init ──────────────────────────────────────────────────────────
    if args.pretrain_epochs > 0:
        print(f"\n===== FedMD Private init (epochs={args.pretrain_epochs}) =====")
        for client in client_list:
            client.pretrain(epochs=args.pretrain_epochs)

    # ── communication rounds ──────────────────────────────────────────────────
    for glob_iter in range(args.Round):
        print(f"\n\n------------- Round number: {glob_iter} -------------\n\n")

        sampled_clients_idx = Sample_Client(args.num_clients, sample_rate=args.sample_rate)
        selected_clients = [client_list[i] for i in sampled_clients_idx]

        # ── randomly sample alignment data for this round (matches FedTED) ────
        try:
            alignment_imgs, _ = next(alignment_iter)
        except StopIteration:
            alignment_iter = iter(alignment_sampler)
            alignment_imgs, _ = next(alignment_iter)
        round_loader = DataLoader(TensorDataset(alignment_imgs), batch_size=args.batch_size, shuffle=False)

        # ── 1. selected clients upload logits on alignment data ───────────────
        all_logits = [client.compute_logits(round_loader) for client in selected_clients]

        # ── 2. server: unweighted mean across selected clients ────────────────
        mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (N, C)

        # ── 3. digest: MAE alignment with mean logits ─────────────────────────
        for client in selected_clients:
            client.digest(round_loader, mean_logits)

        # ── 4. revisit: local training on private data ───────────────────────
        for client in selected_clients:
            client.train()

        result = evaluate_all(client_list, Client_Data, args, glob_iter + 1,
                              recorder=recorder, model_list=model_list)

    return result
