
import gc
import torch

def communication_Pretrain(args, server_model, models, client_weights):
    with torch.no_grad():
        server_state = server_model.state_dict()
        client_states = [model.state_dict() for model in models]

        for key in server_state.keys():
            if 'auxiliary' in key:
                continue

            if 'num_batches_tracked' in key:
                server_state[key].copy_(client_states[0][key])
            else:
                # Efficient weighted sum aggregation
                temp = sum(w * client_states[i][key] for i, w in enumerate(client_weights))

                server_state[key].copy_(temp)

        # Load updated state_dict back into client models
        for model in models:
            model.load_state_dict(server_state)

    return server_model, models

def compute_global_statistic(args, mean_clients, cov_clients, number_clients):
    D = mean_clients.shape[-1]
    total_samples = number_clients.sum(0, keepdim=True)
    mean_weights = number_clients / total_samples.float()
    
    weighted_means = mean_clients * mean_weights.unsqueeze(2).expand(-1, -1, D)
    global_mean = weighted_means.sum(0)
    
    if (total_samples > 1).all():
        cov_weight1 = (number_clients - 1) / (total_samples - 1).float()
        cov_weight2 = number_clients / (total_samples - 1).float()
        cov_weight3 = total_samples / (total_samples - 1).float()
    else:
        cov_weight1 = number_clients / (total_samples + 1e-9).float()
        cov_weight2 = number_clients / (total_samples + 1e-9).float()
        cov_weight3 = total_samples / (total_samples + 1e-9).float()
    
    cov_term1 = (cov_clients * cov_weight1.unsqueeze(2).unsqueeze(3)).sum(0)
    cov_term2 = torch.einsum('abcd, abde->abce', mean_clients.unsqueeze(3), mean_clients.unsqueeze(2))
    cov_term2 = (cov_term2 * cov_weight2.unsqueeze(2).unsqueeze(3)).sum(0)
    cov_term3 = torch.einsum('abc, acd->abd', global_mean.unsqueeze(2), global_mean.unsqueeze(1))
    cov_term3 = cov_term3 * cov_weight3.permute(1, 0).unsqueeze(2)
    
    global_cov = cov_term1 + cov_term2 - cov_term3
    identity_matrix = torch.eye(D).expand(global_cov.shape[0], D, D)
    global_cov += 0.0001 * identity_matrix
    
    unknown_distributions = []
    for idx in range(args.known_class):
        if total_samples[0, idx] > 10:
            unknown_distributions.append(
                torch.distributions.MultivariateNormal(global_mean[idx], covariance_matrix=global_cov[idx])
            )
        else:
            unknown_distributions.append(None)
    
    del cov_term1, cov_term2, cov_term3, cov_weight1, cov_weight2, cov_weight3
    del global_cov, global_mean, identity_matrix
    gc.collect()
    
    return unknown_distributions

def communication_Finetune(args, server_model, models, client_weights, mean_clients, cov_clients, number_clients, unknown_dis):
    if mean_clients:
        mean_clients = torch.stack(mean_clients)    
        cov_clients = torch.stack(cov_clients)    
        number_clients = torch.stack(number_clients) 
        unknown_dis = compute_global_statistic(args, mean_clients, cov_clients, number_clients)

    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'auxiliary' not in key:
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].copy_(models[0].state_dict()[key])
                else:
                    aggregated_param = sum(client_weights[i] * models[i].state_dict()[key] for i in range(len(client_weights)))
                    server_model.state_dict()[key].copy_(aggregated_param)
                    for model in models:
                        model.state_dict()[key].copy_(server_model.state_dict()[key])

    return server_model, models, unknown_dis

