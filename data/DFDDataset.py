"""
The Dataset class for the DFD image data.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import RandomAffine
from data.augmentation import flip, rotate
from pathlib import Path
from PIL import Image
from glob import glob

HQ_SIZE = 128
LQ_SIZE = 32
# hyperparams for the affine transform
DEGREES = [-10, 10]
TRANSLATE = [0.1, 0.1]
SCALE_RANGES = [0.8, 1.2]
SHEARS = None
IMG_SIZE = [32, 32]
MODE = T.InterpolationMode.BILINEAR


class DFDDataset(Dataset):

    def __init__(self, ipath, augment=False, use_ref=True, use_hr_as_ref=False) -> None:
        """
        ipath: Path where the hq and lq images are located.
        augment: Whether to use data augmentation.
        use_ref: whether to use ref images.
        use_hr_as_ref: whether to use the hr image as the ref image.
        """
        self.ipath = ipath
        self.augment = augment
        self.use_ref = use_ref  # not using this right now
        self.use_hr_as_ref = use_hr_as_ref  # not using this right now
        self.hq_paths = sorted(glob(f"{ipath}/hq/*"))
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hq_paths)
    
    def __getitem__(self, index):
        hq_path = self.hq_paths[index]
        img_name = Path(hq_path).stem[:-5]

        # get the hq and lq image
        hq = Image.open(hq_path).resize((HQ_SIZE, HQ_SIZE), resample=Image.BICUBIC)
        # lq = hq.resize((LQ_SIZE, LQ_SIZE), resample=Image.BICUBIC)
        # lq = Image.open(f'{self.ipath}/lq/{Path(hq_path).stem}.png')

        if self.augment:
            pr = T.RandomAffine.get_params(degrees=[-5, 5], 
                                        translate=[0.02, 0.02], 
                                        scale_ranges=[0.95, 1.05], 
                                        shears=None, 
                                        img_size=[128, 128])
            hq = F.affine(hq, *pr, interpolation=Image.BICUBIC)
        
        lq = hq.resize((LQ_SIZE, LQ_SIZE), resample=Image.BICUBIC)

        
        # converting to pytorch tensors
        hq = self.transform(hq)
        lq = self.transform(lq)

        # get the rf image and downsample it
        refs = []
        if self.use_ref:
            for i in range(1, 4):
                rf = Image.open(f"{self.ipath}/ref/{img_name}/{i}.png")
                rf = rf.resize((HQ_SIZE, HQ_SIZE), resample=Image.BICUBIC)
                if self.augment:
                    rf = F.affine(rf, *pr, interpolation=Image.BICUBIC)
                # rfd = rf.resize((LQ_SIZE, LQ_SIZE), resample=Image.BICUBIC)
                # refs.append(self.transform(rfd))
                rf = rf.convert('YCbCr').getchannel('Y')
                rf = self.transform(rf)
                refs.append(rf)

        # data augmentations
        if self.augment:
            hq, lq, refs = flip(hq, lq, refs)
            hq, lq, refs = rotate(hq, lq, refs)

        return hq, lq, refs


class DFDDatasetAlign(Dataset):

    def __init__(self, ipath, augment=False) -> None:
        """
        ipath: Path where the hq and lq images are located.
        augment: Whether to perform rotation and flipping augmentation.
        """
        self.ipath = ipath
        self.augment = augment
        self.raw_paths = sorted(glob(f"{ipath}/raw/*"))
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.raw_paths)
    
    def __getitem__(self, index):
        raw_path = self.raw_paths[index]
        img_name = Path(raw_path).stem[:-5]

        # get the hq and lq image
        raw = Image.open(raw_path)
        ref = Image.open(f"{self.ipath}/ref/{img_name}.png")
        
        # converting to pytorch tensors
        raw = self.transform(raw)
        refs = [self.transform(ref)]

        # get the center crop (exp 32)
        center = F.center_crop(raw, 96)

        # data augmentations
        if self.augment:
            center, raw, refs = flip(center, raw, refs)
            center, raw, refs = rotate(center, raw, refs)

        return center, raw, refs
    

class DFDDatasetAlign2(Dataset):

    def __init__(self, ipath, augment=False) -> None:
        """
        ipath: Path where the hq and lq images are located.
        augment: Whether to perform rotation and flipping augmentation.
        """
        self.ipath = ipath
        self.augment = augment
        self.raw_paths = sorted(glob(f"{ipath}/hq/*"))
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.raw_paths)
    
    def __getitem__(self, index):
        raw_path = self.raw_paths[index]

        # get the hq and lq image
        raw = Image.open(raw_path).resize((LQ_SIZE, LQ_SIZE), resample=Image.BICUBIC)
        
        # converting to pytorch tensors
        raw = self.transform(raw)

        # data augmentations
        dummy = torch.rand(3, 2, 2)
        if self.augment:
            raw, dummy, refs = flip(raw, dummy, refs=None)
            raw, dummy, refs = rotate(raw, dummy, refs=None)

        # generate parameters 
        pr = RandomAffine.get_params(degrees=DEGREES, translate=TRANSLATE, 
                                    scale_ranges=SCALE_RANGES, shears=SHEARS, 
                                    img_size=IMG_SIZE)
        # getting the inverse transform matrix theta and saving
        center=[0.0, 0.0]
        angle = pr[0]
        translate = [t / (IMG_SIZE[0] // 2) for t in pr[1]] 
        scale = pr[2]
        shear = pr[3]
        # unfortunately the server version of torchvision cannot calculate inverse
        # so we manually do it
        theta = F._get_inverse_affine_matrix(center=center, angle=angle,
                                    translate=translate, scale=scale, shear=shear)
        theta = torch.tensor(theta).reshape(2, 3)
        theta_inv_man = theta[..., :-1].inverse() 
        theta_inv_man = torch.cat((theta_inv_man, -theta_inv_man @ theta[..., -1:]), dim=-1)
        
        # creating the transformed image
        refs = [F.affine(raw, *pr, interpolation=MODE)]

        return theta_inv_man, raw, refs

