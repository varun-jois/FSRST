"""
This file contains all the spatial transformers we used
for alignment experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


class STAlign32(nn.Module):
    """
    Designed for 32x32 input
    This block aligns the reference and lr features together
    using a spatial transformer.
    """

    def __init__(self, nc) -> None:
        super().__init__()
        self.nc = nc

        # the localization network designed for 32x32 input
        self.conv = nn.Sequential(
            nn.Conv2d(6, nc, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * nc, nc),
            nn.ReLU(True),
            nn.Linear(nc, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, lq, refs):
        """
        lq: lq features
        ref: ref features
        """
        # concatenate lq and rfd
        ref = refs[0]
        x = torch.cat((lq, ref), 1)

        # pass it through the conv layers
        x = self.conv(x)

        # reshape it so that it can be passed to the fc layers
        x = x.view(-1, 16 * 16 * self.nc)
        theta = self.fc(x)
        
        # perform the sampling of rf (not rfd) based on the theta parameters
        theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, ref.size(), align_corners=False)
        # ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')

        return theta

class STAlign128(nn.Module):
    """
    Designed for 128x128 input
    This block aligns the reference and lr features together
    using a spatial transformer.
    """

    def __init__(self, nc) -> None:
        super().__init__()

        # the localization network designed for 128x128 input
        self.conv = nn.Sequential(
            nn.Conv2d(6, nc, kernel_size=5, stride=1, padding=2), # 128
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(nc, nc, kernel_size=5, stride=1, padding=2), # 64
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(nc, nc, kernel_size=5, stride=1, padding=2), # 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(nc, nc, kernel_size=5, stride=1, padding=2), # 16
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(16 * 16 * nc, nc),
            # nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(nc, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, lq, refs):
        """
        lq: lq features
        refs: ref features which is stored in a list to fit the training loop methods.
        """
        # concatenate lq and rfd
        ref = refs[0]
        x = torch.cat((lq, ref), 1)

        # pass it through the conv layers
        x = self.conv(x)

        # reshape it so that it can be passed to the fc layers
        x = x.view(-1, 16 * 16 * 32)
        theta = self.fc(x)
        
        # perform the sampling of rf (not rfd) based on the theta parameters
        theta = theta.view(-1, 2, 3)
        # for exp 32
        grid = F.affine_grid(theta, ref.size(), align_corners=False)
        ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')
        # ref_a = center_crop(ref_a, 96)

        return ref_a
