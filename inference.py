import torch
import pathlib
import yaml
import numpy
import random
from argparse import ArgumentParser
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader 
from data.RefDataset import RefDatasetSR, RefDatasetAlign
from utils.utils import model_load
from glob import glob

# get the arguments
parser = ArgumentParser()
parser.add_argument('-c', '--cfg', required=True, help='The config yaml.')
args = parser.parse_args()


# load the config file
with open(args.cfg, 'r') as f:
    config = yaml.safe_load(f)
pth = config['paths']
mdl = config['model']
thp = config['train']


# the device
device = torch.device(thp['device'])

# storing some params
checkpt_path = f"{pth['checkpoints']}/{config['name']}"
# to select the checkpoint, we either use the best model, or the model saved at the end
if 'best_mdl.pth' in glob(f"{checkpt_path}/*"):
    cp_num = 'best'
else:
    cp_num = config['thp']['epochs']

# path for saving the results
save_path = f"inference/{config['name']}"


# create the folder for hr and sr files
pathlib.Path(f"{save_path}").mkdir(parents=True, exist_ok=True)


# creating the dataset
if config['dataset'] == 'RefDatasetSR':
    valid_data = RefDatasetSR(pth['valid'], augment=False)
else:
    valid_data = RefDatasetAlign(pth['valid'], augment=False)

# creating the dataloader
valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

# creating the model
if config["model_name"] == 'FSRST':
    from models.FSRST import FSRST as Net
elif config["model_name"] == 'STAlign128':
    from models.STAlign import STAlign128 as Net
elif config["model_name"] == 'STAlign32':
    from models.STAlign import STAlign32 as Net
else:
    raise ValueError('Invalid model name selected.')

# load the model
model = nn.DataParallel(Net(**mdl)).to(device)
model_load({'mdl': model}, cp_num, checkpt_path)

# the main loop
model.eval()
with torch.no_grad():
    for n, data in enumerate(valid_dataloader):
        # load onto device
        hq, lq, refs = data
        hq, lq = hq.to(device), lq.to(device)
        if refs:
            for i in range(len(refs)):
                refs[i] = refs[i].to(device)
            pred = model(lq, refs)
        else:
            pred = model(lq)
        # save images
        save_image(pred[0], f"{save_path}/{n:05d}.png")
        print(f'Image {n} complete')
