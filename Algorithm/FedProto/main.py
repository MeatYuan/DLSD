import torch
import torch.nn as nn
from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.FedProto.Client import Client
from Algorithm.FedProto.default_args import apply_fedproto_defaults
from utils.network import Sample_Client


def _add_latent_layer(model, latent_dim, num_classes):
    """Insert Linear(latent_dim)→ReLU before the final classifier layer.
    Creates a shared proto_dim space across heterogeneous architectures."""
    if isinstance(model.classifier, nn.Linear):
        feature_dim = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.Linear(latent_dim, num_classes),
        )
    elif isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier.children())
        last = layers[-1]
        model.classifier = nn.Sequential(
            *layers[:-1],
            nn.Linear(last.in_features, latent_dim),
            nn.Linear(latent_dim, last.out_features),
        )


def FedProto(Client_Data, models, model_list, args):
    """
    FedProto: Federated Prototype Learning across Heterogeneous Clients.
    Reference: Tan et al., AAAI 2022.

    Prototype = classifier[:-1](feature) in shared latent space (proto_dim).
    Latent layer added to all models so heterogeneous architectures share
    the same prototype space. Per-label aggregation as in original paper.
    """
    apply_fedproto_defaults(args)
    recorder = MetricsRecorder(args)

    for m in models:
        _add_latent_layer(m, args.proto_dim, args.num_classes)

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

    global_protos = {}  # {label: [averaged proto tensor (cpu)]}

    for glob_iter in range(args.Round):
        print(f"\n\n------------- Round number: {glob_iter} -------------\n\n")

        sampled_clients_idx = Sample_Client(args.num_clients, sample_rate=args.sample_rate)
        selected_clients = [client_list[i] for i in sampled_clients_idx]
        # ── client local training with global prototypes ──────────────────────
        local_protos_list = {}
        for client in selected_clients:
            agg_protos_label = client.train(global_protos)
            local_protos_list[client.get_id()] = agg_protos_label

        # ── server: aggregate prototypes (copied from utils.py) ───────────────
        new_protos = proto_aggregation(local_protos_list)
        # retain previous-round protos for classes not seen this round
        for label in global_protos:
            if label not in new_protos:
                new_protos[label] = global_protos[label]
        global_protos = new_protos

        result = evaluate_all(client_list, Client_Data, args, glob_iter + 1,
                              recorder=recorder, model_list=model_list)
    return result


def proto_aggregation(local_protos_list):
    """
    Copied from FedProto/lib/utils.py proto_aggregation().
    Averages per-class prototype tensors across all clients and samples.
    """
    agg_protos_label = {}
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos:
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for label, proto_list in agg_protos_label.items():
        # proto_list[i] is a list of per-sample proto tensors from client i
        all_tensors = []
        for per_client in proto_list:
            all_tensors.extend(per_client)
        proto = torch.stack(all_tensors, dim=0).mean(dim=0)
        agg_protos_label[label] = [proto]

    return agg_protos_label


