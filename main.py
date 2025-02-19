import argparse
import random
import torch
import logging
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # For reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_stage_run(mode: str):
    if mode == 'Pretrain':
        from Fedoss.pretrain import run
    elif mode == 'Finetune':
        from Fedoss.finetune import run
    else:
        raise ValueError(f"Invalid Mode {mode}. Choose 'Pretrain' or 'Finetune'")
    return run

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch Training for Federated Open Set Recognition')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--model_type', default='softmax', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='Resnet18', type=str, help='Backbone type')
    parser.add_argument('--dataset', default='Cifar10', type=str, help='Dataset configuration')
    parser.add_argument('--known_class', default=5, type=int, help='Number of known classes')
    parser.add_argument('--unknown_class', default=3, type=int, help='Number of unknown classes')
    parser.add_argument('--seed', default=47, type=int, help='Random seed for dataset generation')
    parser.add_argument('--data_root', default='./dataset/', type=str, help='Data root path')
    parser.add_argument('--rotation', default=45, type=int, help='Rotation angle')
    parser.add_argument('--resize', default=144, type=int, help='Resize dimension')
    parser.add_argument('--cropsize', default=128, type=int, help='Crop size')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--epoches', default=200, type=int, help='Number of epochs')
    parser.add_argument('--num_client', type=int, default=10, help='Number of clients')
    parser.add_argument('--worker_steps', type=int, default=1, help='Steps of workers')
    parser.add_argument('--mode', type=str, default='Pretrain', help='Training mode: Pretrain or Finetune')
    parser.add_argument('--dirichlet', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--eps', type=float, default=1., help='Epsilon for attack')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of steps for attack')
    parser.add_argument('--unknown_weight', type=float, default=1., help='Weight for unknown classes')
    parser.add_argument('--start_epoch', type=str, default='[5, 10, 15, 20, 25]', help='Epochs to start from')
    parser.add_argument('--sample_from', type=int, default=8, help='Sample from')
    return parser.parse_args()

def main():
    # Start logging
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Starting at {now}")

    args = parse_args()

    # Set up random seed for reproducibility
    set_seed(args.seed)

    # # Create the save path for model outputs
    # args.save_path = create_save_path(args)

    # # Log the arguments for clarity
    # pprint(vars(args))

    # Select the correct function based on the mode (Pretrain/Finetune)
    run_stage = load_stage_run(args.mode)

    # Execute the selected stage
    run_stage(args)

    # Log the ending time
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Ending at {now}")

    
if __name__=='__main__':
    main()


