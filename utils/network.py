import numpy as np
def Sample_Client(num_clients, sample_rate = 1, clients=None, init_seed = None):
    m = max(int(num_clients * sample_rate), 1)
    if init_seed is not None:
        Sample_Client.rng = np.random.default_rng(init_seed)
        print("Client network random seed : ", init_seed)
        return 
    
    sampled_clients_idx = np.sort(Sample_Client.rng.choice(num_clients, size = m, replace=False))

    if clients is not None:
        network_info = {}
        for client_id in sampled_clients_idx:
            neighbor_client = {}
            for neighbor_id in sampled_clients_idx:
                if neighbor_id != client_id:
                    neighbor_client[neighbor_id] = clients[neighbor_id]
            network_info[client_id] = {'Neighbor': neighbor_client}
        return network_info, sampled_clients_idx

    return sampled_clients_idx