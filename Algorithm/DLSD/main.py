from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.DLSD.Client import Client
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
        
        total_global_loss = 0
        total_global_acc = 0
        total_klloss = 0
        total_local_loss = 0
        total_mseloss = 0
        total_ada_loss = 0
        total_ada_ce_loss = 0
        total_ada_align_loss = 0

        for client in client_list:
            if client.get_id() not in sampled_clients_idx:
                continue
            global_loss, global_acc, global_klloss, local_loss, global_mseloss, ada_loss, ada_ce_loss, ada_align_loss = client.train_with_global_dataset(args)
            total_global_loss += global_loss
            total_global_acc += global_acc
            total_klloss += global_klloss
            total_local_loss += local_loss
            total_mseloss += global_mseloss
            total_ada_loss += ada_loss
            total_ada_ce_loss += ada_ce_loss
            total_ada_align_loss += ada_align_loss

            print(f'Round {Round + 1}: Client {client.get_id()} - Global Acc: {global_acc:.4f}, Loss: {global_loss:.4f}, KL Loss: {global_klloss:.4f}, Local Loss: {local_loss:.4f}, MSE Loss: {global_mseloss:.4f}')
            print(f'Round {Round + 1}: Client {client.get_id()} - Adapter Loss: {ada_loss:.4f}, CE Loss: {ada_ce_loss:.4f}, Align Loss: {ada_align_loss:.4f}')
        # Averaged over the clients that actually trained this round, not over all of
        # them -- with sample_rate < 1 the latter scales every training metric down by
        # the participation rate.
        n_trained = max(1, len(sampled_clients_idx))
        print(f'Round {Round + 1}: Global Avg - Train Acc: {total_global_acc/n_trained:.4f}, Loss: {total_global_loss/n_trained:.4f}, KL Loss: {total_klloss/n_trained:.4f}, Local Loss: {total_local_loss/n_trained:.4f}, MSE Loss: {total_mseloss/n_trained:.4f}')
        print(f'Round {Round + 1}: Global Avg - Adapter Loss: {total_ada_loss/n_trained:.4f}, CE Loss: {total_ada_ce_loss/n_trained:.4f}, Align Loss: {total_ada_align_loss/n_trained:.4f}')

        result = evaluate_all(client_list, Client_Data, args, Round + 1,
                              recorder=recorder, model_list=model_list)

    return result




    

