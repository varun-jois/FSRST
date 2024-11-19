"""
A place for all the custom blocks we're building.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.basicblock as B
from torchvision.ops import DeformConv2d


# for making the weights of the offset convolution zero
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)


class FeatureExtractor(nn.Module):
    """
    Create features for the lq input frames.
    It takes in the lq/hq frame and outputs features.
    The design: Conv + ResBlock * 3 + Conv
    """

    def __init__(self, nb, in_nc=3, out_nc=3, nc=64, mode='BRCBRC') -> None:
        super(FeatureExtractor, self).__init__()

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode=mode) for _ in range(nb)]
        m_tail = B.conv(nc, out_nc, mode='C')
        
        self.model = B.sequential(m_head, B.sequential(*m_body), m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class DeformConvAlign(nn.Module):
    """
    The alignment module that aligns using deformable convolutions. 
    Using the unmodulated version.
    Involves two steps: 
    1) offsets = Convolutions([F_lq, F_ref]) 
    2) F_ref_align = DeformableConvolution(F_ref, offsets)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(DeformConvAlign, self).__init__()
        
        # create a convolutions and initialize weights to zeros
        self.conv = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 18, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(init_weights)

        # Initialize the Deformable Convolution layer
        self.deform_conv = DeformConv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, lq, ref):
        """
        lq: lq features
        ref: ref features
        """
        # calculate the offsets
        offsets = self.conv(torch.cat((ref, lq), 1)) # (batch, 2 * kernel^2, h, w)

        # align and perform relu
        ref_a = self.deform_conv(ref, offsets)
        ref_a = self.relu(ref_a)
        return ref_a  


class FeatureAggregation(nn.Module):
    """
    Performs the feature aggregation step so that information is taken in
    a weighted fashion from all the aligned references.
    """

    def __init__(self, in_nc=3, out_nc=3) -> None:
        super(FeatureAggregation, self).__init__()
        self.lq_res = B.ResBlock(in_nc, out_nc)
        self.ref_res = B.ResBlock(in_nc, out_nc)

    def forward(self, lq_f, refs_a):
        """
        lq_f: lq_features
        refs_a: A list of the aligned references
        """
        # get similarity scores
        sim_scores = []
        for ref in refs_a:
            r = torch.transpose(self.ref_res(ref), 2, 3)
            l = self.lq_res(lq_f)
            sim = torch.sigmoid(torch.matmul(r, l))
            sim_scores.append(sim)

        # get the aggregated features
        num = sum(s * r for s, r in zip(sim_scores, refs_a)) # not sure if this is * or @
        den = sum(s for s in sim_scores)
        feature_agg = num / den
        return feature_agg + lq_f

class FrameReconstruction(nn.Module):
    """
    The final part of the model that produces the final, enhanced frame.
    """

    def __init__(self, nb, in_nc=3, out_nc=3, nc=64, mode='BRCBRC'):
        super(FrameReconstruction, self).__init__()
        
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode=mode) for _ in range(nb)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=False)
        
        self.model = B.sequential(m_head, B.sequential(*m_body), m_tail)

    def forward(self, feat_agg):
        """
        Takes in the aggregated features.
        """
        return self.model(feat_agg)
