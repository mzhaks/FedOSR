import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from data.FedCifar10_simple import get_dataloaders
from .common import setup, update_lr
from .communication import communication_Finetune
import os
import random
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu' )
from .synthesis import Attack
import os.path as osp
import gc


def train(args, device, epoch, model, trainloader, optimizer, net_peers=None, attack= None, unknown_dis = None):
    model.train()
    device = DEVICE

    for peer_net in net_peers:
        peer_net.eval()

    train_loss = 0
    pred, label, output_list = [], [], []
    criterion = torch.nn.CrossEntropyLoss()

    net_peers_sample_number = args.num_client - 1 # all clients except for itself

    # Cifar10
    p_lower = 0
    p_upper = 10./16  # Hyperparameter for peer agreement

    unknown_dict = [None for i in range(args.num_client)]
    mean_dict = [None for i in range(args.known_class)]
    cov_dict = [None for i in range(args.known_class)]

    number_dict = torch.zeros(args.known_class)

    for batch_id, (img, labels) in enumerate(trainloader):
        gc.collect()
        torch.cuda.empty_cache()
        img, labels = img.to(device), labels.to(device)
        model = model.to(device)
        outputs  = model(img)
        output = outputs['outputs']
        aux_output = outputs['aux_out']
        boundary_feats = outputs['boundary_feats'] 
        discrete_feats = outputs['discrete_feats']
        loss = criterion(output, labels)
        loss += criterion(aux_output, labels)
        if epoch>=0:
            # Client-Inconcictencies based boundary Samples recognition
            net_peers_sample = random.sample(net_peers, net_peers_sample_number)
            _, aux_pred = aux_output.max(1)   
            aux_preds_peers = torch.eq(aux_pred, labels).float()  
            # print(aux_preds_peers)
            assert len(net_peers)== (args.num_client-1)
            for idx, peer_net in enumerate(net_peers_sample):
                with torch.no_grad():
                    outs_peer = peer_net.aux_forward(boundary_feats.clone().detach())
                    aux_out_peer = outs_peer['aux_out']   
                    _, aux_pred_peer = aux_out_peer.max(1)  
                    aux_preds_peers += torch.eq(aux_pred_peer, labels).float()
            is_boundary_upper = torch.lt(aux_preds_peers/(net_peers_sample_number+1), p_upper)
            is_boundary_lower = torch.gt(aux_preds_peers/(net_peers_sample_number+1), p_lower)
            is_boundary = is_boundary_lower & is_boundary_upper 

            if is_boundary.sum() > 0:
                discrete_feats = discrete_feats[is_boundary]      
                discrete_targets = labels[is_boundary]                 
                inputs_unknown, targets_unknown = attack.i_DUS(model, discrete_feats, discrete_targets) 
                if inputs_unknown is not None:                                        
                    outs_unknown = model.discrete_forward(inputs_unknown.clone().detach()) 
                    outputs_unknown = outs_unknown['outputs']  
                    # probabilistic distance
                    prob_unknown = torch.softmax(outputs_unknown,dim=-1)   
                    PDs = prob_unknown[:,-1] - prob_unknown[:,:-1].max(-1)[0]                    
                    gt_unknown=torch.ones(outputs_unknown.shape[0]).long().to(device)*args.known_class                
                    for i in range(len(outputs_unknown)):
                        nowlabel=targets_unknown[i]
                        outputs_unknown[i][nowlabel]=-1e9                 
                    loss += criterion(outputs_unknown, gt_unknown) * args.unknown_weight   

                    if epoch in args.start_epoch:
                        targets_unknown_numpy = targets_unknown.cpu().data.numpy() 
                        for index in range(len(targets_unknown)):
                            if (PDs[index]>0):
                                dict_key = targets_unknown_numpy[index]
                                unknown_sample = inputs_unknown[index].clone().detach().view(1, -1) 
                                if unknown_dict[dict_key] == None:
                                    unknown_dict[dict_key] = unknown_sample
                                else:
                                    unknown_dict[dict_key] = torch.cat((unknown_dict[dict_key], unknown_sample),dim=0)
                    
                    if unknown_dis is not None:
                        sample_c = torch.randint(0, args.known_class, (args.sample_from,))
                        sample_num = {index: 0 for index in range(args.known_class)}
                        for it in sample_c:
                            sample_num[it.item()] = sample_num[it.item()] + 1 
                        ood_samples = None
                        ood_targets = None
                        for index in range(args.known_class):
                            if sample_num[index] > 0 and unknown_dis[index] != None:                            
                                generated_unknown_samples = unknown_dis[index].rsample((100,))
                                prob_density = unknown_dis[index].log_prob(generated_unknown_samples)
                                # keep the data in the low density area.
                                _, index_prob = torch.topk(- prob_density, sample_num[index])
                                generated_unknown_samples = generated_unknown_samples[index_prob].to(device)
                                generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 192, 4, 4)
                                generated_unknown_targets = (torch.ones(sample_num[index])*index).long().to(device) 
                                if ood_samples is None:
                                    ood_samples = generated_unknown_samples
                                    ood_targets = generated_unknown_targets
                                else:
                                    ood_samples = torch.cat((ood_samples, generated_unknown_samples), 0) 
                                    ood_targets = torch.cat((ood_targets, generated_unknown_targets), 0)
                                del generated_unknown_samples
                        if ood_samples is not None and ood_samples.shape[0]>1:        
                            outs_unknown = model.discrete_forward(ood_samples.clone().detach())      # model.discrete_forward(ood_samples.clone().detach()) 
                            outputs_unknown = outs_unknown['outputs'] 
                            gt_unknown=torch.ones(outputs_unknown.shape[0]).long().to(device)*args.known_class                
                            for i in range(len(outputs_unknown)):
                                nowlabel=ood_targets[i]
                                outputs_unknown[i][nowlabel]=-1e9                 
                            loss += criterion(outputs_unknown, gt_unknown) * args.unknown_weight


        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        value , indices = output[:,:args.known_class].max(1)

        pred.extend(indices.cpu().numpy().tolist())
        label.extend(labels.cpu().numpy().tolist())

        del img, loss
        gc.collect()
    # eps= 1e-3  # epsilon to make cov_matrix positive definite
    if epoch in args.start_epoch: 
        for index in range(args.known_class):
            if unknown_dict[index] is not None:
                mean_dict[index] = unknown_dict[index].mean(0).cpu()
                X = unknown_dict[index] - unknown_dict[index].mean(0)
                X =  X.double()
                # Compute covariance matrix
                cov_matrix = torch.mm(X.t(), X) / len(X)
                # cov_matrix += torch.eye(cov_matrix.shape[0], cov_matrix.device())*eps
                cov_dict[index] = cov_matrix.float().cpu()
                number_dict[index] = len(X)

                del cov_matrix, X
                torch.cuda.empty_cache()
                # mean_dict[index] = unknown_dict[index].mean(0).cpu()  

                # X = unknown_dict[index] - mean_dict[index].to(unknown_dict[index].device)   # Broadcasting
                # X = X.to(dtype=torch.float16)   #(is_boundary.sum(), 4096)
                # # print(X.shape)
                # chunk_size = 512
                # cov_matrix = torch.zeros((X.shape[1], X.shape[1]), device=X.device, dtype=X.dtype)

                # for i in range(0, X.shape[1], chunk_size):
                #     for j in range(0, X.shape[1], chunk_size):
                #         X_chunk_i = X[:, i:i+chunk_size]
                #         X_chunk_j = X[:, j:j+chunk_size]
                #         cov_matrix[i:i+chunk_size, j:j+chunk_size] = torch.mm(X_chunk_i.t(), X_chunk_j) / len(X)  # (4096, 4096)  


                #     del X_chunk_i,  X_chunk_j 
                #     torch.cuda.empty_cache()

                # cov_dict[index] = cov_matrix.cpu().pin_memory()  # Moving to cpu only after computing the cov_matrix
                # number_dict[index] = len(X)   # List of size args.known_class

                # del X, cov_matrix
                # gc.collect()
                # torch.cuda.empty_cache()


                               
            else:
                for i in range(args.known_class):   
                    if unknown_dict[i] is not None:
                       break
                D = unknown_dict[i].shape[1]   
                mean_dict[index] = torch.zeros(D) 
                cov_dict[index] = torch.zeros(D, D)                          
        del unknown_dict
        gc.collect()
        
        mean_dict = torch.stack(mean_dict, dim = 0)   

        cov_dict = torch.stack(cov_dict, dim = 0)    

    for peer_net in net_peers:        
        peer_net.train()          

    loss_avg = train_loss/(batch_id+1)
    mean_acc = 100*metrics.accuracy_score(label, pred)
    precision = 100*metrics.precision_score(label, pred, average='macro', zero_division=0)    
    recall_macro = 100*metrics.recall_score(y_true=label, y_pred=pred, average='macro', zero_division=0)      
    f1_macro = 100*metrics.f1_score(y_true=label, y_pred=pred, average='macro', zero_division=0)    

    result = {'loss':loss_avg,
              'acc':mean_acc,
              'f1': f1_macro,
              'recall':recall_macro,
              'precision': precision,
              'mean_dict' : mean_dict,        
              'cov_dict' : cov_dict,            
              'number_dict' : number_dict       
              }
    
    return result


def val(args, epoch, model, valloader):
    model.eval()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    val_loss = 0
    pred, label,  = [], []
    with torch.no_grad():
        for batch_id, (img, labels) in enumerate(valloader):
            img, labels = img.to(device), labels.to(device)
            model.to(device)
            outputs  = model(img)
            output = outputs['outputs']
            aux_output = outputs['aux_out']
            loss = criterion(output, labels)
            loss += criterion(aux_output, labels)
            val_loss += loss.item()
            value , indices = output[:,:args.known_class].max(1)

            pred.extend(indices.cpu().numpy().tolist())
            label.extend(labels.cpu().numpy().tolist())
        loss_avg = val_loss/(batch_id+1)
        mean_acc = 100*metrics.accuracy_score(label, pred)
        precision = 100*metrics.precision_score(label, pred, average='macro', zero_division=0)    
        recall_macro = 100*metrics.recall_score(y_true=label, y_pred=pred, average='macro',zero_division=0)      
        f1_macro = 100*metrics.f1_score(y_true=label, y_pred=pred, average='macro', zero_division=0)    

        result = {'loss':loss_avg,
                'acc':mean_acc,
                'f1': f1_macro,
                'recall':recall_macro,
                'precision': precision,
                }
    
    return result

def test(args, device, epoch, net, closerloader, openloader, threshold=0):
    net.eval()
    
    temperature = 1.
    with torch.no_grad():
        pred_list=[]
        targets_list=[]
        test_loss=0
        criterion = nn.CrossEntropyLoss()
        
        pred_list_temp = []
        label_list_temp = []
        
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            net = net.to(device)
            outs = net(inputs)
            outputs = outs['outputs']    
            aux_outputs = outs['aux_out']
            loss = criterion(outputs, targets)        
            loss += criterion(aux_outputs, targets)       
            test_loss += loss.item()
            _, predicted = outputs[:, :args.known_class].max(1)
            pred_list_temp.extend(predicted.cpu().numpy().tolist())
            label_list_temp.extend(targets.cpu().numpy().tolist())    

        loss_avg = test_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list_temp, pred_list_temp)
        precision = 100*metrics.precision_score(label_list_temp, pred_list_temp, average='macro', zero_division=0)          
        recall_macro = 100*metrics.recall_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro', zero_division=0)      
        f1_macro = 100*metrics.f1_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro',zero_division=0)    
        confusion_matrix = metrics.confusion_matrix(y_true=label_list_temp, y_pred=pred_list_temp)   
        
        close_test_result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'confusion_matrix':confusion_matrix}        
        
        prob_total = None
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            if prob_total == None:
                prob_total = prob
            else:
                prob_total = torch.cat([prob_total, prob])
            targets_list.append(targets.cpu().numpy())
        
        for batch_idx, (inputs, targets) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            prob_total = torch.cat([prob_total, prob])
            
            targets = np.ones_like(targets.cpu().numpy())*args.known_class
            targets_list.append(targets)

        # openset recognition    
        targets_list=np.reshape(np.array(targets_list),(-1))
        _, pred_list = prob_total.max(1)          
        pred_list = pred_list.cpu().numpy()

        mean_acc = 100.0 * metrics.accuracy_score(targets_list, pred_list)
        precision = 100*metrics.precision_score(targets_list, pred_list, average='macro',zero_division=0)                  
        recall_macro = 100.0*metrics.recall_score(y_true=targets_list, y_pred=pred_list, average='macro', zero_division=0)      
        f1_macro = 100*metrics.f1_score(y_true=targets_list, y_pred=pred_list, average='macro', zero_division=0)    
        confusion_matrix = metrics.confusion_matrix(y_true=targets_list, y_pred=pred_list)
                        
        osr_result = {'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'confusion_matrix': confusion_matrix}

    return osr_result, close_test_result

def initialize(args):
    """Initialize data loaders, models, device, and attack instance."""
    trainloaders, valloader, closerloader, openloader = get_dataloaders(
        args.num_client, args.data_root, 42)
    server_model, models, device, client_weights = setup(args, trainloaders)
    attack = Attack(known_class=args.known_class, eps=args.eps, num_steps=args.num_steps)
    return trainloaders, closerloader, openloader, server_model, models, device, client_weights, attack

def train_clients(args, epoch, device, models, trainloaders, optimizers, attack, unknown_dis):
    """Train clients and collect statistics."""
    mean_clients, cov_clients, number_clients = [], [], []
    
    for client_idx in range(args.num_client):
        client_name = f"Client_{client_idx+1}"
        model, train_loader, optimizer = models[client_idx], trainloaders[client_idx], optimizers[client_idx]
        
        train_result = train(args, device, epoch, model, train_loader, optimizer, 
                             models[:client_idx] + models[client_idx+1:], attack, unknown_dis)
    
        
        print(f"Train {client_name} [{epoch}/{args.epoches}] LR={args.lr:.7f} "
              f"loss={train_result['loss']:.3f} ACC={train_result['acc']:.3f} "
              f"F1={train_result['f1']:.3f} Rec={train_result['recall']:.3f} "
              f"Prec={train_result['precision']:.3f}")

        if epoch in args.start_epoch:
            mean_clients.append(train_result['mean_dict'])
            cov_clients.append(train_result['cov_dict'])
            number_clients.append(train_result['number_dict'])
    
    return mean_clients, cov_clients, number_clients    

def evaluate_and_save(args, epoch, device, server_model, closerloader, openloader, best_f1, best_epoch):
    """Evaluate the model and save the best checkpoint."""
    osr_result, close_test_result = test(args, device, epoch, server_model, closerloader, openloader)
    osr_f1 = osr_result['f1']

    print(f"Test-  OSR [{epoch}/{args.epoches}] LR={args.lr:.7f} "
          f"ACC={osr_result['acc']:.3f} F1={osr_f1:.3f} "
          f"Rec={osr_result['recall']:.3f} Prec={osr_result['precision']:.3f}")
    print(f"Test-Close [{epoch}/{args.epoches}] LR={args.lr:.7f} "
          f"loss={close_test_result['loss']:.3f} ACC={close_test_result['acc']:.3f} "
          f"F1={close_test_result['f1']:.3f} Rec={close_test_result['recall']:.3f} "
          f"Prec={close_test_result['precision']:.3f}")

    if osr_f1 > best_f1:
        best_epoch, best_f1 = epoch, osr_f1
        save_checkpoint(args, server_model, osr_result, best_epoch)
    
    return best_f1, best_epoch

def save_checkpoint(args, server_model, osr_result, best_epoch):
    """Save the best model checkpoint."""
    state = {
        'net': server_model.state_dict(),
        'osr_acc': osr_result['acc'],
        'osr_f1': osr_result['f1'],
        'osr_recall': osr_result['recall'],
        'osr_precision': osr_result['precision'],
        'epoch': best_epoch,
    }
    torch.save(state, osp.join(args.save_path, f'best_finetune_ckpt_{args.mode}.pth'))
    print('Saving best model...')

def main_training_loop(args):
    """Main training loop."""
    (trainloaders, closerloader, openloader, server_model, models, device, 
     client_weights, attack) = initialize(args)
    
    best_f1, best_epoch = 0, 0
    unknown_dis = None
    
    for epoch_it in range(args.epoches // args.worker_steps):
        optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr, 
                                       betas=(0.9, 0.99), amsgrad=False) 
                      for idx in range(args.num_client)]
        
        for ws in range(args.worker_steps):
            mean_clients, cov_clients, number_clients = train_clients(
                args, epoch_it, device, models, trainloaders, optimizers, attack, unknown_dis)    
        
        server_model, models, unknown_dis = communication_Finetune(
            args, server_model, models, client_weights, mean_clients, cov_clients, number_clients, unknown_dis)
        
        if mean_clients:
            del mean_clients, cov_clients, number_clients
            gc.collect()
        
        best_f1, best_epoch = evaluate_and_save(args, epoch_it, device, server_model, closerloader, openloader, best_f1, best_epoch)
    
    print('------> Best performance ----->>>>>>')
    print(f'Best Epoch: {best_epoch}, Best OSR F1: {best_f1:.3f}')
    print('==================================')



def run(args):
    main_training_loop(args)

    


if __name__== '__main__':
    # import sklearn
    pass



    
