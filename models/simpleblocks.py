import torch
import torch.nn as nn

class SimpleAggregation(nn.Module):
    """
    A simple aggregation model for data where we have an lq image with a facial
    feature missing and a ref image with just that facial feature. Can this
    model learn to take info from the reference image?
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # 1x1 convolutions for lq and rf image
        self.conv_lq = nn.Conv2d(6, 3, 1)
        self.conv_rf = nn.Conv2d(6, 3, 1)

    def forward(self, lq, rf):
        """
        Takes one lq image with the feature (like nose or eyes) masked out and
        the ref image which has this feature.
        lq: [b, c, h, w]
        rf: [b, c, h, w] (same as lq)
        """
        merged = torch.cat([lq, rf], 1)
        lq = self.conv_lq(merged)
        rf = self.conv_rf(merged)
        return lq + rf
