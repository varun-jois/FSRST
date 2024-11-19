import torch.nn as nn
import torch.nn.functional as F

class Bicubic(nn.Module):

    def __init__(self, output_size) -> None:
        super(Bicubic, self).__init__()
        self.output_size = output_size

    def forward(self, lq):
        result =  F.interpolate(lq, (self.output_size, self.output_size), 
                                mode='bicubic', align_corners=False)
        return result
