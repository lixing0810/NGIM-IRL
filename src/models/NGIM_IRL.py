"""
5-Stage Dual Network for Low-Dose CT Reconstruction.

This model implements a physics-informed deep learning approach for CT reconstruction,
combining iterative reconstruction with learned denoising in both sinogram and image domains.
The model uses 5 stages of iterative refinement with UNet-based denoising and learnable
step size parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Import projection operators
try:
    import odl
    from odl.contrib import torch as odl_torch

    ODL_AVAILABLE = True
except ImportError:
    ODL_AVAILABLE = False
    print("Warning: ODL not available. Projection operators will need to be provided.")

from .unet import UNet


class BypassRound(torch.autograd.Function):
    """
    Custom rounding function with straight-through gradient estimator.

    This function implements differentiable rounding for photon counting statistics.
    The backward pass uses the straight-through estimator.
    """

    @staticmethod
    def forward(ctx, inputs):
        """Apply rounding operation."""
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through gradient estimator."""
        return grad_output


class ProjectionOperators:
    """
    Manages forward and backward projection operators for CT reconstruction.
    """

    @staticmethod
    def build_lodopab_geometry():
        """
        Build CT geometry configuration for LoDoPaB dataset.

        Returns:
            odl.Operator: Ray transform operator
        """
        if not ODL_AVAILABLE:
            raise ImportError("ODL is required to build geometry. Please install odl.")

        # Physical constants
        MU_WATER = 20
        MU_AIR = 0.02

        # Image geometry: ~26cm x 26cm images
        MIN_PT = [-0.13, -0.13]
        MAX_PT = [0.13, 0.13]
        NUM_ANGLES = 1000
        RECO_IM_SHAPE = (362, 362)

        # Create reconstruction space
        space = odl.uniform_discr(
            min_pt=MIN_PT,
            max_pt=MAX_PT,
            shape=RECO_IM_SHAPE,
            dtype=np.float32
        )

        # Create geometry
        geometry = odl.tomo.parallel_beam_geometry(
            space,
            num_angles=NUM_ANGLES
        )

        # Create ray transform operator
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        return ray_trafo

    @classmethod
    def create_operators(cls, device='cuda'):
        """
        Create forward and backward projection operators.

        Args:
            device (str): Device to place operators on

        Returns:
            tuple: (forward_operator, backward_operator)
        """
        if not ODL_AVAILABLE:
            raise ImportError("ODL is required to create operators.")

        ray_trafo = cls.build_lodopab_geometry()
        forward_operator = odl_torch.OperatorModule(ray_trafo)
        backward_operator = odl_torch.OperatorModule(ray_trafo.adjoint)

        # Move to device if possible
        if hasattr(forward_operator, 'to'):
            forward_operator = forward_operator.to(device)
            backward_operator = backward_operator.to(device)

        return forward_operator, backward_operator


class DualNetStage(nn.Module):  # 改为继承 nn.Module
    """
    Represents a single stage of the DualNet architecture.

    This class encapsulates the learnable parameters and update equations
    for one stage of the iterative reconstruction.
    """

    def __init__(self, stage_idx, device='cuda'):
        """
        Initialize a stage with learnable parameters.

        Args:
            stage_idx (int): Stage index (1-5)
            device (str): Device to place parameters on
        """
        super(DualNetStage, self).__init__()  # 调用父类初始化

        self.stage_idx = stage_idx

        # Initialize learnable parameters
        self.eta1 = nn.Parameter(torch.tensor([0.01], device=device))
        self.eta2 = nn.Parameter(torch.tensor([0.01], device=device))
        self.eta3 = nn.Parameter(torch.tensor([0.01], device=device))

        # Store device
        self.device = device

    def update_q(self, q_value, p_org, incident_flux, y_value, sigma2):
        """
        Update quantized photon counts (Q).

        Args:
            q_value: Current quantized photon counts
            p_org: Original photon counts
            incident_flux: Incident X-ray flux
            y_value: Current sinogram estimate
            sigma2: Noise variance parameter

        Returns:
            Updated quantized photon counts
        """
        # Euler-Mascheroni constant
        euler_gamma = 0.57722

        # Compute update: q = q - η1 * ((q - p)/σ² - log(I₀) + y + log(q) + γ)
        update_term = (q_value - p_org) / sigma2
        update_term = update_term - torch.log(incident_flux)
        update_term = update_term + y_value + torch.log(q_value) + euler_gamma

        q_updated = q_value - self.eta1 * update_term

        # Apply rounding with straight-through gradient
        return BypassRound.apply(q_updated)

    def update_y(self, q_value, incident_flux, y_value, x_value, forward_operator, prox_net_y):
        """
        Update sinogram estimate (Y).

        Args:
            q_value: Current quantized photon counts
            incident_flux: Incident X-ray flux
            y_value: Current sinogram estimate
            x_value: Current image estimate
            forward_operator: Forward projection operator
            prox_net_y: UNet for sinogram denoising

        Returns:
            Updated sinogram estimate
        """
        # Compute update: y = y - η2 * ((q - I₀*exp(-y))/I₀ - A(x) + y)
        exp_term = incident_flux * torch.exp(-y_value)
        q_update = (q_value - exp_term) / incident_flux
        y_update = q_update - forward_operator(x_value) + y_value

        y_updated = y_value - self.eta2 * y_update

        # Apply UNet denoising
        return prox_net_y(y_updated)

    def update_x(self, y_value, x_value, forward_operator, backward_operator, prox_net_x):
        """
        Update image estimate (X).

        Args:
            y_value: Current sinogram estimate
            x_value: Current image estimate
            forward_operator: Forward projection operator
            backward_operator: Backward projection operator
            prox_net_x: UNet for image denoising

        Returns:
            Updated image estimate
        """
        # Compute update: x = x - η3 * Aᵀ(A(x) - y)
        forward_x = forward_operator(x_value)
        x_update = backward_operator(forward_x - y_value)

        x_updated = x_value - self.eta3 * x_update

        # Apply UNet denoising
        return prox_net_x(x_updated)


class NGIM_IRL(nn.Module):
    """
    5-Stage Dual Network for CT Reconstruction.

    This model implements a physics-informed deep learning approach for
    low-dose CT reconstruction using iterative refinement with learned
    denoising operators.

    The model alternates between:
    1. Updating quantized photon counts with Poisson statistics
    2. Updating sinogram estimates with forward projection
    3. Updating image estimates with back projection

    Each update step includes UNet-based denoising and learnable step sizes.
    """

    def __init__(self, device='cuda', unet_chans=64, unet_depth=4, dropout=0.0):
        """
        Initialize the 5-stage dual network.

        Args:
            device (str): Device to place model on ('cuda' or 'cpu')
            unet_chans (int): Number of channels in UNet
            unet_depth (int): Depth of UNet
            dropout (float): Dropout probability
        """
        super(NGIM_IRL, self).__init__()

        self.device = torch.device(device)
        self.num_stages = 5

        # Create projection operators
        self.forward_operator, self.backward_operator = self._create_projection_operators()

        # Create UNets for each stage
        self.unets_y = nn.ModuleList([
            UNet(1, 1, unet_chans, unet_depth, dropout) for _ in range(self.num_stages)
        ])

        self.unets_x = nn.ModuleList([
            UNet(1, 1, unet_chans, unet_depth, dropout) for _ in range(self.num_stages)
        ])

        # Create stages with learnable parameters - 现在 DualNetStage 继承自 nn.Module
        self.stages = nn.ModuleList([
            DualNetStage(i + 1, device=device) for i in range(self.num_stages)
        ])

        # Noise variance parameter (σ²)
        self.sigma2 = nn.Parameter(torch.tensor([11.0], device=device))

        # Initialize parameters
        self._initialize_parameters()

        # Move model to device
        self.to(self.device)

        print(f"Initialized DualNet5Stage with {self.num_stages} stages")
        print(f"  Device: {device}")
        print(f"  Sigma² (noise variance): {self.sigma2.item():.4f}")
        for i, stage in enumerate(self.stages):
            print(f"  Stage {i + 1}: η1={stage.eta1.item():.4f}, "
                  f"η2={stage.eta2.item():.4f}, η3={stage.eta3.item():.4f}")

    def _create_projection_operators(self):
        """
        Create forward and backward projection operators.

        Returns:
            tuple: (forward_operator, backward_operator)
        """
        if ODL_AVAILABLE:
            try:
                return ProjectionOperators.create_operators(device=self.device)
            except Exception as e:
                print(f"Warning: Could not create ODL operators: {e}")
                print("Using placeholder operators. Provide custom operators for actual use.")

        # Return placeholder functions if ODL is not available
        def placeholder_forward(x):
            return x  # Identity function as placeholder

        def placeholder_backward(x):
            return x  # Identity function as placeholder

        return placeholder_forward, placeholder_backward

    def _initialize_parameters(self):
        """Initialize model parameters."""
        # UNet parameters are initialized in their __init__
        # Stage parameters are initialized in DualNetStage.__init__
        pass

    def forward(self, y_value, incident_flux, middle_value, x_init):
        """
        Forward pass through the 5-stage network.

        Args:
            y_value (torch.Tensor): Low-dose sinogram measurements
                Shape: [batch_size, 1, num_angles, detector_size]
            incident_flux (torch.Tensor): Incident X-ray flux (I₀)
                Can be scalar or tensor
            middle_value (torch.Tensor): Clipping threshold for positivity
                Shape: [batch_size, 1, num_angles, detector_size]
            x_init (torch.Tensor): Initial FBP reconstruction
                Shape: [batch_size, 1, image_size, image_size]

        Returns:
            torch.Tensor: Reconstructed CT image
                Shape: [batch_size, 1, image_size, image_size]
        """
        # Ensure tensors are on the correct device
        y_value = y_value.to(self.device)
        incident_flux = incident_flux.to(self.device)
        middle_value = middle_value.to(self.device)
        x_init = x_init.to(self.device)

        # Ensure incident_flux has correct shape for broadcasting
        if incident_flux.dim() == 0:
            incident_flux = incident_flux.unsqueeze(0)
        if incident_flux.dim() == 1:
            incident_flux = incident_flux.view(-1, 1, 1, 1)

        # Initial values
        y_current = y_value
        p_org = incident_flux * torch.exp(-y_current)
        q_current = torch.maximum(p_org, middle_value)
        x_current = x_init

        # Iterate through stages
        for i in range(self.num_stages):
            stage = self.stages[i]
            unet_y = self.unets_y[i]
            unet_x = self.unets_x[i]

            # Update quantized photon counts (Q)
            q_current = stage.update_q(
                q_current, p_org, incident_flux, y_current, self.sigma2
            )

            # Update sinogram estimate (Y)
            y_current = stage.update_y(
                q_current, incident_flux, y_current, x_current,
                self.forward_operator, unet_y
            )

            # Update image estimate (X)
            x_current = stage.update_x(
                y_current, x_current, self.forward_operator,
                self.backward_operator, unet_x
            )

        return x_current

    def get_parameters(self):
        """
        Get all learnable parameters.

        Returns:
            dict: Dictionary containing all parameters
        """
        params = {
            'sigma2': self.sigma2
        }

        for i, stage in enumerate(self.stages):
            params[f'stage_{i + 1}_eta1'] = stage.eta1
            params[f'stage_{i + 1}_eta2'] = stage.eta2
            params[f'stage_{i + 1}_eta3'] = stage.eta3

        return params
