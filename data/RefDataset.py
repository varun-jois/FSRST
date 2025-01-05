"""
The Dataset class for the reference based SR dataset.
"""
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from data.augmentation import flip, rotate
from pathlib import Path
from PIL import Image
from glob import glob

HR_SIZE = 128 
LR_SIZE = 32


class RefDatasetSR(Dataset):
    """
    Reference based dataset for super-resolution.
    """

    def __init__(self, ipath, augment=False) -> None:
        """
        ipath: Path where the hr and lr images are located.
        augment: Whether to use data augmentation.
        """
        self.ipath = ipath
        self.augment = augment
        self.hr_paths = sorted(glob(f"{ipath}/hr/*"))
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_paths)
    
    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        img_name = Path(hr_path).stem

        # get the hr and lr images
        hr = Image.open(hr_path).resize((HR_SIZE, HR_SIZE), resample=Image.BICUBIC)

        if self.augment:
            pr = T.RandomAffine.get_params(degrees=[-5, 5], 
                                        translate=[0.02, 0.02], 
                                        scale_ranges=[0.95, 1.05], 
                                        shears=None, 
                                        img_size=[HR_SIZE, HR_SIZE])
            hr = F.affine(hr, *pr, interpolation=Image.BICUBIC)
        
        lr = hr.resize((LR_SIZE, LR_SIZE), resample=Image.BICUBIC)

        
        # converting to pytorch tensors
        hr = self.transform(hr)
        lr = self.transform(lr)

        # get the rf image and make it the same size as hr image
        refs = []
        for i in range(1, 4):
            rf = Image.open(f"{self.ipath}/rf/{img_name}/{i}.png")
            rf = rf.resize((HR_SIZE, HR_SIZE), resample=Image.BICUBIC)
            if self.augment:
                rf = F.affine(rf, *pr, interpolation=Image.BICUBIC)
            rf = rf.convert('YCbCr').getchannel('Y')
            rf = self.transform(rf)
            refs.append(rf)

        # data augmentations
        if self.augment:
            hr, lr, refs = flip(hr, lr, refs)
            hr, lr, refs = rotate(hr, lr, refs)

        return hr, lr, refs


class RefDatasetAlign(Dataset):
    """
    Reference based dataset for alignment.
    """

    def __init__(self, ipath, augment=False) -> None:
        """
        ipath: Path where the image and reference images are located.
        augment: Whether to perform rotation and flipping augmentation.
        """
        self.ipath = ipath
        self.augment = augment
        self.img_paths = sorted(glob(f"{ipath}/im/*"))
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = Path(img_path).stem

        # get the image and reference
        im = Image.open(img_path).resize((HR_SIZE, HR_SIZE), resample=Image.BICUBIC)
        rf = Image.open(f"{self.ipath}/rf/{img_name}.png").resize((HR_SIZE, HR_SIZE), resample=Image.BICUBIC)
        
        # converting to pytorch tensors
        im = self.transform(im)
        refs = [self.transform(rf)]

        # we take a center crop to ignore the border pixels.
        # if you change this, make the change in STAlign.py as well
        # here we are assuming 128x128 pixel images
        center = F.center_crop(im, 124)

        # data augmentations
        if self.augment:
            center, im, refs = flip(center, im, refs)
            center, im, refs = rotate(center, im, refs)

        return center, im, refs
