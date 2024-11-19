"""
File for data augmentations.
"""

import torch
import torchvision.transforms.functional as F

def flip(hq, lq, refs):
    """
    Performs random horizontal and vertical flips.
    """
    rn = torch.randint(low=0, high=4, size=(1,)).item()
    if rn == 1:
        hq = F.hflip(hq)
        lq = F.hflip(lq)
        if refs:
            for i in range(len(refs)):
                refs[i] = F.hflip(refs[i])
    if rn == 2:
        hq = F.vflip(hq)
        lq = F.vflip(lq)
        if refs:
            for i in range(len(refs)):
                refs[i] = F.vflip(refs[i])
    if rn == 3:
        hq = F.hflip(F.vflip(hq))
        lq = F.hflip(F.vflip(lq))  
        if refs:     
            for i in range(len(refs)):
                refs[i] = F.hflip(F.vflip(refs[i]))
    return hq, lq, refs

def rotate(hq, lq, refs):
    """
    Performs random 90 degree rotations.
    """
    rn = torch.randint(low=0, high=4, size=(1,)).item()
    if rn == 1:
        hq = F.rotate(hq, 90)
        lq = F.rotate(lq, 90)
        if refs:
            for i in range(len(refs)):
                refs[i] = F.rotate(refs[i], 90)
    if rn == 2:
        hq = F.rotate(hq, 180)
        lq = F.rotate(lq, 180)
        if refs:
            for i in range(len(refs)):
                refs[i] = F.rotate(refs[i], 180)
    if rn == 3:
        hq = F.rotate(hq, 270)
        lq = F.rotate(lq, 270) 
        if refs:     
            for i in range(len(refs)):
                refs[i] = F.rotate(refs[i], 270)
    return hq, lq, refs
