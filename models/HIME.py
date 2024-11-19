
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
        m_body = [B.ResBlock(nc, nc, mode='BRCBRC') for _ in range(nb)]
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

    def __init__(self, in_nc=3, out_nc=3, nc=32):
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
        # chan = in_nc + out_nc
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(chan, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(chan, nc, 3, padding=1),
        #     nn.BatchNorm2d(nc, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nc, 18, 3, padding=1)
        # )
        
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


class FrameReconstruction(nn.Module):
    """
    The final part of the model that produces the final, enhanced frame.
    """

    def __init__(self, in_nc=3, out_nc=3, nb=20, nc=32):
        super(FrameReconstruction, self).__init__()
        
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='BRCBRC') for _ in range(nb)]
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
        m_body_1 = [B.ResBlock(nc1, nc1, mode='BRCBRC') for _ in range(nb1)]
        
        transition = B.conv(nc1, nc2, mode='C')

        # part 2
        m_body_2 = [B.ResBlock(nc2, nc2, mode='BRCBRC') for _ in range(nb2)]
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


class HIME(nn.Module):

    def __init__(self, nc) -> None:
        super(HIME, self).__init__()

        # pixelunshuffle for the ref images
        self.unshuffle = nn.PixelUnshuffle(4)
        
        # the feature extractors
        self.feature_extraction_lq = FeatureExtractor(nb=5, nc=nc, out_nc=3)
        self.feature_extraction_ref = FeatureExtractor(in_nc=16, nb=3, nc=nc, out_nc=3)   

        # the feature alignment block
        self.feature_align = DeformConvAlign(in_nc=3, out_nc=3, nc=nc)

        # the feature aggregation block
        self.feature_aggregation = FeatureAggregation(in_nc=3, out_nc=3)

        # the reconstruction block and shuffle for exp 36
        # self.reconstruction = FrameReconstruction(in_nc=3, out_nc=nc, nc=nc, nb=20)
        # # pixel shuffle with icnr initialization
        # self.shuffle = PixelShuffle_ICNR(nc, 48)

        # the reconstruction block and shuffle for exp 84
        self.reconstruction = FrameReconstruction2(out_nc=48, nc1=nc, nc2=nc)
        # pixel shuffle
        self.shuffle = nn.PixelShuffle(4)


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

        return lq + pred