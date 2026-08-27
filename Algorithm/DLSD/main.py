from Algorithm.base import MetricsRecorder, Losses, evaluate_all
from Algorithm.DLSD.Client import Client, TRAIN_METRICS
from utils.network import Sample_Client


def DLSD(Client_Data, models, model_list, args):
    recorder = MetricsRecorder(args)

    client_list = [Client(model = models[i],
                          local_train_loader = Client_Data["client_train"][i], 
                          local_test_loader = Client_Data["client_test"], 
                          client_id = i, 
                          args = args,
                          device=args.device) for i in range(args.num_clients)]
    
    for client in client_list:
        client.Generate_Syn_Data(args)

    #Train local model with GAN
    for Round in range(args.Round):

        network_info, sampled_clients_idx = Sample_Client(args.num_clients, sample_rate=args.sample_rate, clients = client_list)

        for client_id in sampled_clients_idx:
            client_list[client_id].get_features_dataset(args, rounds=Round)

        for client_id in sampled_clients_idx:
            client_list[client_id].Init_neighbors(network_info[client_id]['Neighbor'])

        for client in sampled_clients_idx:
            client_list[client].transmit_info(adapter_agg = args.Adapter_agg)
        
        if args.Adapter_agg:
            for client in sampled_clients_idx:
                client_list[client].aggergate_adapter()
        
        # Averaged over the clients that actually trained this round, not over all of
        # them -- with sample_rate < 1 the latter scales every training metric down by
        # the participation rate.
        round_avg = Losses(*TRAIN_METRICS)

        for client in client_list:
            if client.get_id() not in sampled_clients_idx:
                continue
            m = client.train_with_global_dataset(args)
            round_avg.merge(m)

            print(f'Round {Round + 1}: Client {client.get_id()} - Global Acc: {m["acc"]:.4f}, Loss: {m["loss"]:.4f}, KL Loss: {m["kd"]:.4f}, Local Loss: {m["local"]:.4f}, MSE Loss: {m["mse"]:.4f}')
            print(f'Round {Round + 1}: Client {client.get_id()} - Adapter Loss: {m["ada"]:.4f}, CE Loss: {m["ada_ce"]:.4f}, Align Loss: {m["ada_align"]:.4f}')

        print(f'Round {Round + 1}: Global Avg - Train Acc: {round_avg["acc"]:.4f}, Loss: {round_avg["loss"]:.4f}, KL Loss: {round_avg["kd"]:.4f}, Local Loss: {round_avg["local"]:.4f}, MSE Loss: {round_avg["mse"]:.4f}')
        print(f'Round {Round + 1}: Global Avg - Adapter Loss: {round_avg["ada"]:.4f}, CE Loss: {round_avg["ada_ce"]:.4f}, Align Loss: {round_avg["ada_align"]:.4f}')

        result = evaluate_all(client_list, Client_Data, args, Round + 1,
                              recorder=recorder, model_list=model_list)

    return result




    

