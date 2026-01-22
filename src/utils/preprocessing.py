"""
Preprocessing utilities for CT reconstruction.

This module contains functions for data normalization, augmentation,
and preprocessing specific to CT reconstruction tasks.
"""

import numpy as np
import torch
import odl

def normalize_image(image, min_val=0.0, max_val=1.0):
    """
    Normalize image to [0, 255] range.

    Args:
        image (np.ndarray): Input image
        min_val (float): Minimum value for clipping
        max_val (float): Maximum value for clipping

    Returns:
        np.ndarray: Normalized image
    """
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    image = image.astype(np.float32) * 255.0
    return image


def image_get_minmax():
    """Get default min/max values for CT images."""
    return 0.0, 1.0


def proj_get_minmax():
    """Get default min/max values for projection data."""
    return 0.0, 4.0


def preprocess_sinogram(sinogram, use_fbp=True):
    """
    Preprocess sinogram data.

    Args:
        sinogram (np.ndarray): Input sinogram
        use_fbp (bool): Whether to apply FBP for initial reconstruction

    Returns:
        tuple: (processed_sinogram, fbp_reconstruction if use_fbp else None)
    """
    # Normalize sinogram
    sinogram_normalized = normalize_image(
        sinogram,
        *proj_get_minmax()
    )

    # Apply FBP if requested
    if use_fbp:
        fbp_recon = fbp(sinogram)
        return sinogram_normalized, fbp_recon
    else:
        return sinogram_normalized, None


def create_batch_tensors(low_dose_data, normal_dose_data, device='cuda'):
    """
    Create batched tensors for training.

    Args:
        low_dose_data (np.ndarray): Low-dose sinograms
        normal_dose_data (np.ndarray): Normal-dose reconstructions
        device (str): Device to place tensors on

    Returns:
        tuple: (low_dose_tensor, normal_dose_tensor)
    """
    # Convert to torch tensors
    low_dose_tensor = torch.FloatTensor(low_dose_data).unsqueeze(1)
    normal_dose_tensor = torch.FloatTensor(normal_dose_data).unsqueeze(1)

    # Move to device
    if device:
        low_dose_tensor = low_dose_tensor.to(device)
        normal_dose_tensor = normal_dose_tensor.to(device)

    return low_dose_tensor, normal_dose_tensor


def compute_data_statistics(data_loader):
    """
    Compute mean and std of dataset for normalization.

    Args:
        data_loader: DataLoader for the dataset

    Returns:
        dict: Statistics including mean and std
    """
    means = []
    stds = []

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            data = batch[0]  # Assume first element is input data
        else:
            data = batch

        means.append(data.mean().item())
        stds.append(data.std().item())

    return {
        'mean': np.mean(means),
        'std': np.mean(stds),
        'num_samples': len(data_loader.dataset)
    }


def fbp(sino):
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
