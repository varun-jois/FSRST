
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
    The design: Conv + ResBlock * nb + Conv
    """

    def __init__(self, in_nc=3, out_nc=3, nc=32, nb=3) -> None:
        super(FeatureExtractor, self).__init__()

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='RCRC') for _ in range(nb)] # 'BRCBRC'
        m_tail = B.conv(nc, out_nc, mode='C')
        
        self.model = B.sequential(m_head, B.sequential(*m_body), m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class SpatialTransformerAlign(nn.Module):
    """
    This block aligns the reference and lr features together
    using a spatial transformer.
    """

    def __init__(self, lr) -> None:
        super().__init__()

        # the localization network designed for 32x32 input
        if lr == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True)
            )
        elif lr == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True)
            )
        else:
            raise ValueError('lr for SpatialTransformerAlign should be 16 or 32')
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, lq, ref):
        """
        lq: lq features
        ref: ref features
        """
        # concatenate lq and rfd
        x = torch.cat((lq, ref), 1)

        # pass it through the conv layers
        x = self.conv(x)

        # reshape it so that it can be passed to the fc layers
        x = x.view(-1, 16 * 16 * 32)
        theta = self.fc(x)
        
        # perform the sampling of rf (not rfd) based on the theta parameters
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, ref.size(), align_corners=False)
        ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')

        return ref_a
    

class SpatialTransformerAlignParams(nn.Module):
    """
    This block aligns the reference and lr features together
    using a spatial transformer.
    """

    def __init__(self) -> None:
        super().__init__()

        # the localization network designed for 32x32 input
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, lq, ref):
        """
        lq: lq features
        ref: ref features
        """
        # concatenate lq and rfd
        x = torch.cat((lq, ref), 1)

        # pass it through the conv layers
        x = self.conv(x)

        # reshape it so that it can be passed to the fc layers
        x = x.view(-1, 16 * 16 * 32)
        theta = self.fc(x)
        
        # perform the sampling of rf (not rfd) based on the theta parameters
        theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, ref.size(), align_corners=False)
        # ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')

        return theta
    
    def align(self, ref, theta):
        """
        Takes the ref and aligns using theta.
        Here the ref is full resolution.
        """
        grid = F.affine_grid(theta, ref.size(), align_corners=False)
        ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')
        return ref_a


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
            nn.BatchNorm2d(6, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 12, 3, padding=1),
            nn.BatchNorm2d(12, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 18, 3, padding=1)
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
        self.lq_res = nn.Conv2d(in_nc, out_nc, 3, padding=1)
        self.ref_res = nn.Conv2d(in_nc, out_nc, 3, padding=1)

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


class FeatureAggregationExp(nn.Module):
    """
    Performs the feature aggregation step so that information is taken in
    a weighted fashion from all the aligned references.

    This uses softmax and a modification of the frobenius norm.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, lq_f, refs_a):
        # get count of refs
        rcount = len(refs_a)
        # get the modified norm distance for the pixel positions
        refs_sub = [lq_f.sub(r).square().sum(dim=1, keepdim=True).sqrt() 
                    for r in refs_a]
        # concatenate the above tensors and clamp to avoid div by 0
        refs_sub = torch.cat(refs_sub, 1).clamp(max=1e2)
        # get the softmax of the negative to prioritize smaller distance from lq
        smax = (refs_sub.neg().exp()) / (refs_sub.neg().exp().sum(dim=1, keepdim=True) + 1e-9)
        # concatenate the given refs
        refs_a = torch.cat(refs_a, 1)
        # multiply the softmax weights to the provided refs
        refs_a = refs_a.mul(smax.repeat_interleave(rcount, dim=1))
        # aggregate all the weigthed refs
        refs_a = sum(refs_a.tensor_split(rcount, dim=1))
        return refs_a + lq_f
        # return torch.cat([lq_f, refs_a], 1)


# class FeatureAggregationWT(nn.Module):
#     """
#     Performs the feature aggregation using a simple convolution block.
#     """

#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, lq_f, refs_a):
#         pass


class FrameReconstruction(nn.Module):
    """
    The final part of the model that produces the final, enhanced frame.
    """

    def __init__(self, in_nc=3, out_nc=3, nb=20, nc=32):
        super(FrameReconstruction, self).__init__()
        
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='RCRC') for _ in range(nb)]  # 'BRCBRC'
        m_tail = B.conv(nc, out_nc, mode='C', bias=False)

        self.model = B.sequential(m_head, B.sequential(*m_body), m_tail)

    def forward(self, feat_agg):
        """
        Takes in the aggregated features.
        """
        return self.model(feat_agg)


class FrameReconstruction2(nn.Module):
    """
    The final part of the model that produces the final, enhanced frame.
    """

    def __init__(self, in_nc=3, out_nc=3, nb1=8, nb2=12, nc1=32, nc2=48):
        super(FrameReconstruction2, self).__init__()
        
        # part 1
        m_head = B.conv(in_nc, nc1, mode='C')
        m_body_1 = [B.ResBlock(nc1, nc1, mode='RCRC') for _ in range(nb1)] # 'BRCBRC'
        
        transition = B.conv(nc1, nc2, mode='C')

        # part 2
        m_body_2 = [B.ResBlock(nc2, nc2, mode='RCRC') for _ in range(nb2)]  # 'BRCBRC'
        m_tail = B.conv(nc2, out_nc, mode='C', bias=False)

        
        self.model = B.sequential(m_head, m_body_1, B.sequential(*m_body_1), transition,
                                  B.sequential(*m_body_2), m_tail)

    def forward(self, feat_agg):
        """
        Takes in the aggregated features.
        """
        return self.model(feat_agg)


# icnr initialization for pixelshuffle
# taken from https://github.com/fastai/fastai/blob/master/fastai/layers.py
def icnr_init(x, scale=4, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    # code I added to follow the random distribution for Conv2d
    bound = (1 / (nf * h * w)) ** 0.5
    k = nn.init.uniform_(x.new_zeros([ni2,nf,h,w]), -bound, bound).transpose(0, 1)
    # k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf):
        super().__init__()
        layers = [nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                  nn.PixelShuffle(4)]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        layers[0].bias.data.fill_(0) # I added this
        super().__init__(*layers)

    
"""
Original HIMEv2
"""
class HIME(nn.Module):

    def __init__(self, nc, uf, lr) -> None:
        super(HIME, self).__init__()

        # pixelunshuffle for the ref images
        self.unshuffle = nn.PixelUnshuffle(uf)
        
        # the feature extractors
        self.feature_extraction_lq = FeatureExtractor(nb=5, nc=nc)
        self.feature_extraction_ref = FeatureExtractor(in_nc=int(uf**2), nb=5, nc=nc)   

        # the feature alignment block
        self.feature_align = SpatialTransformerAlign(lr)

        # the feature aggregation block
        self.feature_aggregation = FeatureAggregationExp()

        # the reconstruction block
        # self.reconstruction = FrameReconstruction2(out_nc=48, nc1=nc, nc2=nc)
        self.reconstruction = FrameReconstruction(out_nc=int(3*uf**2), nc=nc, nb=20)

        # pixel shuffle
        self.shuffle = nn.PixelShuffle(uf)

        # # pixel shuffle with icnr initialization
        # self.shuffle = PixelShuffle_ICNR(64, 48)

        # last convolution
        self.last = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, lq, refs):
        """
        lq: The low quality frame.
        refs: The high quality reference frames in a list.
        """
        # perform the unshuffle
        refs_f = [self.unshuffle(r) for r in refs]

        # extract features
        lq_f = self.feature_extraction_lq(lq)
        refs_f = [self.feature_extraction_ref(r) for r in refs_f]

        # align ref features
        refs_f = [self.feature_align(lq_f, r) for r in refs_f]

        # aggregate all features
        feat_agg = self.feature_aggregation(lq_f, refs_f)

        # produce the final output
        pred = self.reconstruction(feat_agg)

        # get back to hq size
        pred = self.shuffle(pred)
        pred = self.last(pred)

        # use bicubic to upscale lq
        out_shape = pred.shape[2]
        lq = F.interpolate(lq, (out_shape, out_shape), mode='bicubic', 
                           align_corners=False).clamp(min=0, max=1)
        # lq = F.interpolate(lq, (out_shape, out_shape), 
        #                    mode='nearest').clamp(min=0, max=1)

        # return lq + pred
        return (lq + pred).clamp(min=0, max=1)

"""
Experimental HIMEv2
"""
# class HIME(nn.Module):

#     def __init__(self, nc) -> None:
#         super(HIME, self).__init__()

#         # pixelunshuffle for the ref images
#         self.unshuffle = nn.PixelUnshuffle(4)
        
#         # the feature extractors
#         self.feature_extraction_lq = FeatureExtractor(nb=5, nc=nc)
#         self.feature_extraction_ref = FeatureExtractor(in_nc=16, nb=5, nc=nc)   

#         # the feature alignment block
#         self.feature_align = SpatialTransformerAlign()

#         # the feature aggregation block
#         self.feature_aggregation = FeatureAggregationExp()

#         # the reconstruction block
#         # self.reconstruction = FrameReconstruction2(out_nc=48, nc1=nc, nc2=nc)
#         self.reconstruction = FrameReconstruction(out_nc=48, nc=nc, nb=20, in_nc=12)

#         # pixel shuffle
#         self.shuffle = nn.PixelShuffle(4)

#         # # pixel shuffle with icnr initialization
#         # self.shuffle = PixelShuffle_ICNR(64, 48)

#         # last convolution
#         self.last = nn.Conv2d(3, 3, 3, padding=1)

#     def forward(self, lq, refs):
#         """
#         lq: The low quality frame.
#         refs: The high quality reference frames in a list.
#         """
#         # perform the unshuffle
#         refs_f = [self.unshuffle(r) for r in refs]

#         # extract features
#         lq_f = self.feature_extraction_lq(lq)
#         refs_f = [self.feature_extraction_ref(r) for r in refs_f]

#         # align ref features
#         refs_f = [self.feature_align(lq_f, r) for r in refs_f]

#         # aggregate all features
#         feat_agg = self.feature_aggregation(lq_f, refs_f)

#         # produce the final output
#         pred = self.reconstruction(feat_agg)

#         # get back to hq size
#         pred = self.shuffle(pred)
#         pred = self.last(pred)

#         # use bicubic to upscale lq
#         out_shape = pred.shape[2]
#         lq = F.interpolate(lq, (out_shape, out_shape), mode='bicubic', 
#                            align_corners=False).clamp(min=0, max=1)

#         # return lq + pred
#         return (lq + pred).clamp(min=0, max=1)
    
"""
HIMEv3 #################################################
"""

# class HIMEv3(nn.Module):

#     def __init__(self, nc) -> None:
#         super(HIMEv3, self).__init__()

#         # pixelunshuffle for the ref images
#         self.unshuffle = nn.PixelUnshuffle(4)
        
#         # the feature extractors
#         self.feature_extraction_lq = FeatureExtractor(nb=5, nc=nc)
#         self.feature_extraction_ref = FeatureExtractor(in_nc=16, nb=5, nc=nc)   

#         # the feature alignment block
#         self.feature_align = SpatialTransformerAlign()

#         # the feature aggregation block
#         self.feature_aggregation = FeatureAggregationExp()

#         # the reconstruction block
#         self.reconstruction = FrameReconstruction2(out_nc=48, nc1=nc, nc2=nc)
#         # self.reconstruction = FrameReconstruction(out_nc=48, nc=nc, nb=20)

#         # pixel shuffle
#         self.shuffle = nn.PixelShuffle(4)

#         # # pixel shuffle with icnr initialization
#         # self.shuffle = PixelShuffle_ICNR(64, 48)

#         # last convolution
#         self.last = nn.Conv2d(3, 3, 3, padding=1)

#     def forward(self, lq, refs):
#         """
#         lq: The low quality frame.
#         refs: The high quality reference frames in a list.
#         """
#         # perform the unshuffle
#         refs_f = [self.unshuffle(r) for r in refs]

#         # extract features
#         lq_f = self.feature_extraction_lq(lq)
#         refs_f = [self.feature_extraction_ref(r) for r in refs_f]

#         # align ref features
#         refs_f = [self.feature_align(lq_f, r) for r in refs_f]

#         # aggregate all features
#         feat_agg = self.feature_aggregation(lq_f, refs_f)

#         # produce the final output
#         pred = self.reconstruction(feat_agg)

#         # get back to hq size
#         pred = self.shuffle(pred)
#         pred = self.last(pred)

#         # use bicubic to upscale lq
#         out_shape = pred.shape[2]
#         lq = F.interpolate(lq, (out_shape, out_shape), mode='bicubic', 
#                            align_corners=False).clamp(min=0, max=1)

#         # return lq + pred
#         return (lq + pred).clamp(min=0, max=1)

"""
Exp 62 - Aligning raw references and their feature representations.
"""
# class HIME(nn.Module):

#     def __init__(self, nc) -> None:
#         super(HIME, self).__init__()

#         self.feature_align_input = SpatialTransformerAlignParams()  # it's a dumb name so that it's also an offset param

#         # pixelunshuffle for the ref images
#         self.unshuffle = nn.PixelUnshuffle(4)
        
#         # the feature extractors
#         self.feature_extraction_lq = FeatureExtractor(nb=5, nc=nc)
#         self.feature_extraction_ref = FeatureExtractor(in_nc=16, nb=3, nc=nc)   

#         # the feature alignment block
#         self.feature_align = SpatialTransformerAlign()

#         # the feature aggregation block
#         self.feature_aggregation = FeatureAggregationExp()

#         # the reconstruction block
#         self.reconstruction = FrameReconstruction(out_nc=48, nc=nc, nb=20)

#         # pixel shuffle
#         self.shuffle = nn.PixelShuffle(4)

#         # # pixel shuffle with icnr initialization
#         # self.shuffle = PixelShuffle_ICNR(64, 48)

#         # last convolution
#         self.last = nn.Conv2d(3, 3, 3, padding=1)

#     def forward(self, lq, refs):
#         """
#         lq: The low quality frame.
#         refs: The high quality reference frames in a list.
#         """
#         # separate the lr and hr refs
#         refs_lr = refs[::2]
#         refs_hr = refs[1::2]
#         assert refs_lr[0].shape[2] == 32
#         assert refs_hr[0].shape[2] == 128
#         # align the refs first
#         for i in range(len(refs_lr)):
#             theta = self.feature_align_input(lq, refs_lr[i])
#             refs_hr[i] = self.feature_align_input.align(refs_hr[i], theta)

#         # perform the unshuffle
#         refs_f = [self.unshuffle(r) for r in refs_hr]

#         # extract features
#         lq_f = self.feature_extraction_lq(lq)
#         refs_f = [self.feature_extraction_ref(r) for r in refs_f]

#         # align ref features
#         refs_f = [self.feature_align(lq_f, r) for r in refs_f]

#         # aggregate all features
#         feat_agg = self.feature_aggregation(lq_f, refs_f)

#         # produce the final output
#         pred = self.reconstruction(feat_agg)

#         # get back to hq size
#         pred = self.shuffle(pred)
#         pred = self.last(pred)

#         # use bicubic to upscale lq
#         out_shape = pred.shape[2]
#         lq = F.interpolate(lq, (out_shape, out_shape), mode='bicubic', 
#                            align_corners=False).clamp(min=0, max=1)

#         # return lq + pred
#         return (lq + pred).clamp(min=0, max=1)
    