#!/usr/bin/env python3
"""
Main training script for CT reconstruction.

This script handles the complete training pipeline including:
- Configuration loading
- Data preparation
- Model initialization
- Training execution
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from ctnet.dataloader.data_lodopab import *
from dival.datasets.lodopab_dataset import LoDoPaBDataset
# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Subset
from public_code.src.models.NGIM_IRL_1 import NGIM_IRL
import ctnet.lodopab.network_round_withq_onechannel as net

from public_code.src.training.trainer import CTReconstructionTrainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config):
    """Setup training environment."""
    # Set CUDA device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(config.get('cuda_device', 0))
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Create output directories
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['result_dir'], exist_ok=True)

    return device


def create_model(config, device):
    """Create model instance."""
    model_config = config['model']
    model = NGIM_IRL()
    # model = net.DualNet_New_5_stage().cuda()
    # Move model to device
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {model_config['name']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def create_optimizer(model, config):
    """Create optimizer."""
    optim_config = config['optimizer']

    if optim_config['type'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=optim_config.get('weight_decay', 0.0),
            betas=(
                optim_config.get('beta1', 0.9),
                optim_config.get('beta2', 0.999)
            )
        )
    elif optim_config['type'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=optim_config.get('momentum', 0.9),
            weight_decay=optim_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_config['type']}")

    return optimizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CT reconstruction model')
    parser.add_argument('--config', type=str, default='/home/lixing/CT_RECONSTRUCTION/public_code/config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root directory')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override data root if specified
    if args.data_root:
        config['data']['data_root'] = args.data_root

    # Setup environment
    device = setup_environment(config)

    # Create data loaders
    print("Creating data loaders...")
    dataset = LoDoPaBDataset(impl='astra_cuda')

    # full_train_set = dataset.create_torch_dataset('train')
    # train_indices = list(range(10))  # 只取前10个样本
    # train_subset = Subset(full_train_set, train_indices)
    train_set = dataset.create_torch_dataset('train')
    train_loader = data.DataLoader(train_set, batch_size= config['training']['batch_size'], shuffle=False, num_workers=0,
                                   pin_memory=False)

    # test_set = dataset.create_torch_dataset('test'),
    full_test_set = dataset.create_torch_dataset('test')
    test_indices = list(range(3553))  # 取前3553个样本
    test_subset = Subset(full_test_set, test_indices)
    val_loader = data.DataLoader(test_subset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=False)


    # Create model
    model = create_model(config, device)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create loss function
    loss_config = config['loss']
    if loss_config['type'].lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_config['type'].lower() == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    # Create trainer
    trainer = CTReconstructionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device
    )

    # Resume training if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(config['training']['epochs'])

    # Cleanup
    trainer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
