import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from data.FedCifar10_simple import get_dataloaders
from .common import setup, update_lr
from .communication import communication_Pretrain
import os
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu' )


def train(args, epoch, model, trainloader):
    model.train()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=8)
    optimizer = torch.optim.Adam(model.parameters())

    train_loss = 0
    pred, label, output_list = [], [], []

    for batch_id, (img, labels) in enumerate(trainloader):
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        model = model.to(device)
        outputs  = model(img)
        output = outputs['outputs']
        aux_output = outputs['aux_out']
        
        loss = criterion(output, labels)
        loss += criterion(aux_output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        value , indices = output[:,:args.known_class].max(1)

        pred.extend(indices.cpu().numpy().tolist())
        label.extend(labels.cpu().numpy().tolist())
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

def evaluate_model(net, dataloader, device, known_classes, criterion, temperature=1.0):
    """Helper function to evaluate the model on a given dataset."""
    net.eval()
    total_loss, pred_list, label_list = 0, [], []
    prob_total, targets_list = None, []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = net(inputs)
            outputs, aux_outputs = outs['outputs'], outs['aux_out']
            
            # Compute loss
            loss = criterion(outputs, targets) + criterion(aux_outputs, targets)
            total_loss += loss.item()
            
            # Predictions and probabilities
            _, predicted = outputs[:, :known_classes].max(1)
            pred_list.extend(predicted.cpu().numpy().tolist())
            label_list.extend(targets.cpu().numpy().tolist())
            
            prob = nn.functional.softmax(outputs / temperature, dim=-1)
            prob_total = prob if prob_total is None else torch.cat([prob_total, prob])
            targets_list.append(targets.cpu().numpy())
            
    # Compute metrics
    loss_avg = total_loss / len(dataloader)
    mean_acc = 100 * metrics.accuracy_score(label_list, pred_list)
    precision = 100 * metrics.precision_score(label_list, pred_list, average='macro',zero_division=0)
    recall = 100 * metrics.recall_score(label_list, pred_list, average='macro', zero_division=0)
    f1 = 100 * metrics.f1_score(label_list, pred_list, average='macro', zero_division=0)
    conf_matrix = metrics.confusion_matrix(label_list, pred_list)
    
    return {
        'loss': loss_avg,
        'acc': mean_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'prob_total': prob_total,
        'targets_list': targets_list,
    }

def test(args, device, epoch, net, closerloader, openloader, threshold=0):
    """Evaluates the model on closed-set and open-set recognition tasks."""
    criterion = nn.CrossEntropyLoss()
    
    # Closed-set evaluation
    close_test_result = evaluate_model(net, closerloader, device, args.known_class, criterion)
    prob_total, targets_list = close_test_result['prob_total'], close_test_result['targets_list']
    
    # Open-set evaluation
    with torch.no_grad():
        for inputs, targets in openloader:
            inputs = inputs.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            
            prob = nn.functional.softmax(outputs, dim=-1)
            prob_total = torch.cat([prob_total, prob])
            
            targets = torch.full_like(targets, args.known_class).cpu().numpy()
            targets_list.append(targets)
    
    # Process open-set predictions
    targets_list = np.concatenate(targets_list, axis=0)
    pred_list = prob_total.argmax(dim=1).cpu().numpy()
    
    mean_acc = 100 * metrics.accuracy_score(targets_list, pred_list)
    precision = 100 * metrics.precision_score(targets_list, pred_list, average='macro', zero_division=0)
    recall = 100 * metrics.recall_score(targets_list, pred_list, average='macro', zero_division=0)
    f1 = 100 * metrics.f1_score(targets_list, pred_list, average='macro', zero_division=0)
    conf_matrix = metrics.confusion_matrix(targets_list, pred_list)
    
    osr_result = {
        'acc': mean_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
    }
    
    return osr_result, close_test_result


def train_federated_model(args):
    # Load datasets and initialize models
    loaders = get_dataloaders(args.num_client, args.data_root, 42, args)
    trainloaders, valloader, closerloader, openloader = loaders
    
    server_model, models, device, client_weights = setup(args, trainloaders)
    best_f1, best_epoch = 0, 0
    
    for epoch in range(args.epoches // args.worker_steps):
        args.lr = update_lr(args.lr, epoch, args.epoches, lr_step=20, lr_gamma=0.5)
        optimizers = [
            torch.optim.Adam(models[idx].parameters(), lr=args.lr, betas=(0.9, 0.99))
            for idx in range(args.num_client)
        ]
        
        for _ in range(args.worker_steps):
            for client_idx, (model, train_loader, optimizer) in enumerate(zip(models, trainloaders, optimizers)):
                client_name = f"Client_{client_idx}"
                train_result = train(args, epoch, model, train_loader)
                log_metrics("Train", client_name, epoch, args.epoches, args.lr, train_result)
        
        # Aggregation step
        server_model, models = communication_Pretrain(args, server_model, models, client_weights)
        
        # Validation phase
        val_result = val(args, epoch, server_model, valloader)
        log_metrics("Val", "Server", epoch, args.epoches, args.lr, val_result)
        
        # Save best model
        if val_result['f1'] > best_f1:
            best_f1, best_epoch = val_result['f1'], epoch
            save_models(args, epoch, server_model, models, closerloader, openloader)
    
    # Print best performance
    print(f"------> Best performance at epoch {best_epoch} <------")

def log_metrics(stage, client_name, epoch, total_epochs, lr, metrics):
    print(
        f"{stage} {client_name} [{epoch}/{total_epochs}] LR={lr:.7f} "
        f"ACC={metrics['acc']:.3f} "
        f"F1={metrics['f1']:.3f} Rec={metrics['recall']:.3f} "
        f"Prec={metrics['precision']:.3f}"
    )

def save_models(args, epoch, server_model, models, closerloader, openloader):
    osr_result, close_test_result = test(args, DEVICE, epoch, server_model, closerloader, openloader)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print(f"Saving best model at epoch {epoch}...")
    torch.save({'net': server_model.state_dict()}, os.path.join(args.save_path, generate_model_name(args, epoch)))
    
    for idx, model in enumerate(models):
        torch.save({'net': model.state_dict()}, os.path.join(args.save_path, generate_model_name(args, epoch, client_idx=idx)))
    
    log_metrics("Test-OSR", "Server", epoch, args.epoches, args.lr, osr_result)
    log_metrics("Test-Close", "Server", epoch, args.epoches, args.lr, close_test_result)

def generate_model_name(args, epoch, client_idx=None):
    name = f"best_ckpt_{args.mode}_known_{args.known_class}_unknown_{args.unknown_class}_seed_{args.seed}"
    if client_idx is not None:
        name += f"_C_{client_idx}"
    return f"{name}.pth"


def run(args):
    train_federated_model(args)

    


if __name__== '__main__':
    # import sklearn
    pass



    