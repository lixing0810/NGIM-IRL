"""
Training module for CT reconstruction.

This module handles the training loop, validation, and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import odl
import torch.nn.functional as F
from kornia.filters import get_gaussian_kernel2d, filter2D

def compute_ssim(img1, img2, window_size=11, reduction: str = "mean", max_val: float = 1.0, full: bool = False):
    window: torch.Tensor = get_gaussian_kernel2d(
        (window_size, window_size), (1.5, 1.5))
    window = window.requires_grad_(False)
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2
    tmp_kernel: torch.Tensor = window.to(img1)
    tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
    # compute local mean per channel
    mu1: torch.Tensor = filter2D(img1, tmp_kernel)
    mu2: torch.Tensor = filter2D(img2, tmp_kernel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
    sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
    sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

    ssim_map = ((2. * mu1_mu2 + C1) * (2. * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map
    if reduction != 'none':
        ssim_score = torch.clamp(ssim_score, min=0, max=1)
        if reduction == "mean":
            ssim_score = torch.mean(ssim_score)
        elif reduction == "sum":
            ssim_score = torch.sum(ssim_score)
    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ssim_score, cs
    return ssim_score


def compute_psnr(input: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    mse_val = F.mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)

def fbp_torch(sino):
    MU_WATER = 20
    MU_AIR = 0.02
    MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

    # ~26cm x 26cm images
    MIN_PT = [-0.13, -0.13]
    MAX_PT = [0.13, 0.13]

    NUM_ANGLES = 1000
    RECO_IM_SHAPE = (362, 362)

    # image shape for simulation
    IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)
    IM_SHAPE = RECO_IM_SHAPE

    reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,
                                   shape=RECO_IM_SHAPE, dtype=np.float32)
    space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                              dtype=np.float32)

    reco_geometry = odl.tomo.parallel_beam_geometry(
        reco_space, num_angles=NUM_ANGLES)
    geometry = odl.tomo.parallel_beam_geometry(
        space, num_angles=NUM_ANGLES, det_shape=reco_geometry.detector.shape)

    IMPL = 'astra_cuda'
    # reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl=IMPL)
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)
    fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)
    return fbp(sino)

class CTReconstructionTrainer:
    """
    Trainer class for CT reconstruction models.

    Args:
        model (nn.Module): The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        config (dict): Configuration dictionary
        device (str): Device to train on ('cuda' or 'cpu')
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device

        # Setup logging
        self.writer = SummaryWriter(config['output']['log_dir'])
        self.checkpoint_dir = config['output']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Constants from physics configuration
        self.incident_flux = torch.Tensor([config['physics']['incident_flux']]).to(device)
        self.middle_value = torch.ones(
            1, 1, config['physics']['num_angles'], config['physics']['detector_size']
        ).to(device) * 0.1

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}') as pbar:
            for batch_idx, (low_dose, normal_dose) in enumerate(pbar):
                # Move data to device
                low_dose = low_dose.to(self.device)
                normal_dose = normal_dose.to(self.device)

                # Generate FBP reconstruction for initialization
                fbp_init = fbp_torch(low_dose.squeeze().cpu().numpy())
                fbp_init = torch.FloatTensor(fbp_init).unsqueeze(0).unsqueeze(0).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                reconstruction = self.model(
                    low_dose,
                    self.incident_flux,
                    self.middle_value,
                    fbp_init
                )

                # Compute loss
                loss = self.criterion(reconstruction, normal_dose)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    recon_np = reconstruction.squeeze()
                    target_np = normal_dose.squeeze()
                    psnr = compute_psnr(recon_np, target_np)

                # Update metrics
                epoch_loss += loss.item()
                epoch_psnr += psnr

                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': psnr,
                    'avg_loss': epoch_loss / (batch_idx + 1),
                    'avg_psnr': epoch_psnr / (batch_idx + 1)
                })

                # Log batch metrics to tensorboard
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/batch_psnr', psnr, global_step)

        # Compute epoch averages
        epoch_loss /= num_batches
        epoch_psnr /= num_batches

        return {
            'loss': epoch_loss,
            'psnr': epoch_psnr
        }

    def validate(self):
        """
        Validate the model.

        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for low_dose, normal_dose in tqdm(self.val_loader, desc='Validation'):
                    # Move data to device
                low_dose = low_dose.to(self.device)
                normal_dose = normal_dose.to(self.device)

                # Generate FBP reconstruction
                fbp_init = fbp_torch(low_dose.squeeze().cpu().numpy())
                fbp_init = torch.FloatTensor(fbp_init).unsqueeze(0).unsqueeze(0).to(self.device)

                # Forward pass
                reconstruction = self.model(
                    low_dose,
                    self.incident_flux,
                    self.middle_value,
                    fbp_init
                )

                # Compute loss and metrics
                loss = self.criterion(reconstruction, normal_dose)
                recon_np = reconstruction
                target_np = normal_dose.unsqueeze(0)

                psnr = compute_psnr(recon_np, target_np)
                ssim = compute_ssim(recon_np, target_np)

                val_loss += loss.item()
                val_psnr += psnr
                val_ssim += ssim

        # Compute averages
        val_loss /= num_batches
        val_psnr /= num_batches
        val_ssim /= num_batches

        return {
            'loss': val_loss,
            'psnr': val_psnr,
            'ssim': val_ssim
        }

    def train(self, num_epochs):
        """
        Main training loop.

        Args:
            num_epochs (int): Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(val_metrics['loss'])

            # Adjust learning rate if needed
            self._adjust_learning_rate()

    def _log_epoch_metrics(self, train_metrics, val_metrics):
        """Log metrics for current epoch."""
        print(f"\nEpoch {self.current_epoch + 1}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f} dB, "
              f"SSIM: {val_metrics['ssim']:.4f}")

        # Log to tensorboard
        self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('epoch/train_psnr', train_metrics['psnr'], self.current_epoch)
        self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('epoch/val_psnr', val_metrics['psnr'], self.current_epoch)
        self.writer.add_scalar('epoch/val_ssim', val_metrics['ssim'], self.current_epoch)

    def _save_checkpoint(self, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch + 1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  Saved best model with loss: {val_loss:.4f}")

    def _adjust_learning_rate(self):
        """Adjust learning rate according to schedule."""
        # Example: Reduce LR by half every 5 epochs
        if (self.current_epoch + 1) % 5 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  Learning rate reduced to {self.optimizer.param_groups[0]['lr']}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def close(self):
        """Cleanup resources."""
        self.writer.close()
