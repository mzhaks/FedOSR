from .model import resnet18, resnet9
import numpy as np
import torch
from copy import deepcopy
import os

def generate_model_name(args, epoch=None, client_idx=None):
    name = f"best_ckpt_{args.mode}_{args.known_class}_unknown_{args.unknown_class}_seed_{args.seed}"
    if client_idx is not None:
        name += f"_C_{client_idx}"
    return f"{name}.pth"


def setup(args, trainloaders):
    device = ('cuda' if torch.cuda.is_available() else 'cpu' )
    
    if args.mode == 'Finetune':
        base = "./pretrained_model"
    
        server_model_path = os.path.join(base, generate_model_name(args))
        server_model_path = os.path.abspath(server_model_path)
        if args.model == 'resnet18':
            server_model = resnet18(num_classes=args.known_class)
        elif args.model == 'resnet9':
            server_model = resnet9(num_classes=args.known_class)
        
        if os.path.exists(server_model_path):
            server_model.load_state_dict(torch.load(server_model_path, map_location=device))
        
        server_model = server_model.to(device)
        
        models = []
        for client_idx in range(args.num_client):
            client_model_path = os.path.join(base, generate_model_name(args, client_idx=client_idx))
            client_model = resnet18(num_classes=args.known_class)

            
            if os.path.exists(client_model_path):
                client_model.load_state_dict(torch.load(client_model_path, map_location=device))
            
            models.append(client_model.to(device))
        
    else:
        server_model = resnet18(num_classes=args.known_class)
        
        models = [deepcopy(server_model).to(device) for idx in range(args.num_client)]

    sample_num = np.array([trainloader.dataset.__len__() for trainloader in trainloaders])
    client_weights = sample_num / sample_num.sum()
    return server_model, models, device,  client_weights


def update_lr(lr, epoch, n_epoch, lr_step=20, lr_gamma=0.5):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    if (epoch + 1) % (n_epoch//4) == 0 and (epoch + 1) != n_epoch:  # Yeah, ugly but will clean that later
        lr *= lr_gamma
        print(f'>> New learning Rate: {lr}')
        
    return lr

