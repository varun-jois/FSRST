import sys
import logging
import torch
import pathlib
import yaml
import numpy
import random
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from utils.utils import train_loop, valid_loop, model_save, model_load
from models.loss import CharbonierLoss
from argparse import ArgumentParser

# torch.autograd.set_detect_anomaly(True)

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


# set the seed for the RNG
seed = thp['seed']
g = torch.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
# g = torch.Generator()
# g.manual_seed(seed)


# the device
device = torch.device(thp['device'])

# storing some params
checkpt_path = f"{pth['checkpoints']}/{config['name']}"
log = f'{checkpt_path}/{config["name"]}.log' 

# create the checkpoints folder and log file
pathlib.Path(checkpt_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(log).touch(exist_ok=True)

# creating summary writer and logging file
file_handler = logging.FileHandler(filename=log)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
writer = SummaryWriter(f'runs/{config["name"]}')


# store the config file in the log
logging.info(config)

# getting data paths
dpath = pth['data']

# creating the dataset
if config['dataset'] == 'CelebA':
    if config["model_name"] == 'STAlign128':
        from data.CelebADataset import CelebADatasetAlign
        train_data = CelebADatasetAlign(pth['train'])
        valid_data = CelebADatasetAlign(pth['valid'])
    else:
        from data.CelebADataset import CelebADataset
        # train_json = f'{dpath}/train.json'
        # valid_json = f'{dpath}/valid.json'
        # train_data = CelebADataset(train_json, pth['train'], hr_size=thp['hr_size'], lr_size=thp['lr_size'], type='train', use_ref=thp['use_ref'], use_hr_as_ref=thp['use_hr_as_ref'])
        # valid_data = CelebADataset(valid_json, pth['valid'], hr_size=thp['hr_size'], lr_size=thp['lr_size'], type='valid', use_ref=thp['use_ref'], use_hr_as_ref=thp['use_hr_as_ref'])
        train_data = CelebADataset('train', config)
        valid_data = CelebADataset('valid', config)
elif config['dataset'] == 'DFD':
    if config["model_name"] in ['STAlign32', 'STAlign128']:
        from data.DFDDataset import DFDDatasetAlign2
        train_data = DFDDatasetAlign2(pth['train'], augment=True)
        valid_data = DFDDatasetAlign2(pth['valid'])
    else:
        from data.DFDDataset import DFDDataset
        train_data = DFDDataset(pth['train'], augment=True, use_ref=thp['use_ref'], use_hr_as_ref=thp['use_hr_as_ref'])
        valid_data = DFDDataset(pth['valid'], augment=False, use_ref=thp['use_ref'], use_hr_as_ref=thp['use_hr_as_ref'])
else:
    raise ValueError('Invalid dataset selected in the config.yaml')

# creating the dataloaders
train_dataloader = DataLoader(train_data, batch_size=thp['batch_size'], shuffle=True, 
                              worker_init_fn=seed_worker, generator=g)
valid_dataloader = DataLoader(valid_data, batch_size=thp['batch_size'], shuffle=True,
                              worker_init_fn=seed_worker, generator=g)

# creating the model
if config["model_name"] == 'HIMEv2':
    from models.HIMEv2 import HIME as Net
elif config["model_name"] == 'HIMEv3':
    from models.HIMEv2 import HIMEv3 as Net
elif config["model_name"] == 'STAlign128':
    from models.STAlign import STAlign128 as Net
elif config["model_name"] == 'STAlign32':
    from models.STAlign import STAlign32 as Net
elif config["model_name"] == 'ResBlock':
    from models.basicblock import ResBlock as Net
elif config["model_name"] == 'ResNetSR':
    from models.ResNetSR import ResNetSR as Net
elif config["model_name"] == 'HIME':
    from models.HIME import HIME as Net
elif config["model_name"][:5] == 'Model':
    from models.Model_1 import Model as Net
else:
    raise ValueError('Invalid model name selected.')

# create the model
model = nn.DataParallel(Net(**mdl)).to(device)
logging.info(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f'Total params: {total_params:,}')

# using a pretrained alignment network that we don't want to update with backprop
# if config['name'] == 'exp_47':
#     sd = torch.load('checkpoints/exp_46/2100_raw_mdl.pth')
#     model.module.feature_align.load_state_dict(sd)
#     for p in model.module.feature_align.parameters():
#         p.requires_grad = False
#     logging.info(f'Loaded the pretrained alignment module.')

# creating parameter groups
# if config["model_name"] == 'HIME':
#     offset_params = ['module.feature_align.conv.0.weight', 'module.feature_align.conv.0.bias',
#                     'module.feature_align.conv.2.weight', 'module.feature_align.conv.2.bias']
#     base_params = [p for n, p in model.named_parameters() if n not in offset_params]
#     offset_params = [p for n, p in model.named_parameters() if n in offset_params]

if config["model_name"].startswith('HIME'):
    base_params = [p for n, p in model.named_parameters() if not n.startswith('module.feature_align') and p.requires_grad]
    logging.info(f'Number of base params: {len(base_params)}')
    offset_params = [p for n, p in model.named_parameters() if n.startswith('module.feature_align') and p.requires_grad]
    logging.info(f'Number of offset params: {len(offset_params)}')


# create optimizer
if config["model_name"].startswith('HIME'):
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': offset_params, 'lr': thp['offset_lr']}
    ], lr=thp['learning_rate'])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=thp['learning_rate'])

model_data = {'mdl': model, 'opt': optimizer}


# creating a scheduler
if thp['scheduler']:
    scheduler = MultiStepLR(optimizer, thp['steps'], thp['gamma'])
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=thp['learning_rate']*.01)
    # model_data['sch'] = scheduler


# load pretrained model
if thp['epoch_start'] != 0:
    model_load(model_data, thp['epoch_start'], checkpt_path)
    # manually setting the lr for the groups if resuming training
    optimizer.param_groups[0]['lr'] = thp['learning_rate']
    if config["model_name"].startswith('HIME'):
        optimizer.param_groups[1]['lr'] = thp['offset_lr']
    if thp['scheduler']:
        scheduler.last_epoch = thp['epoch_start']

# loss functions
losses = {'loss_rec': nn.L1Loss()}

# training loop
best_score = thp['save_thresh']
for epoch in range(1 + thp["epoch_start"], thp["epoch_start"] + thp["epochs"] + 1):
    #print(f"-------------------------------\nEpoch {epoch}")
    logging.info(f"-------------------------------\nEpoch {epoch}")
    
    # iterate over the data
    train_loss = train_loop(train_dataloader, model_data, losses, device)
    writer.add_scalar('loss/train', train_loss, epoch)

    # validation
    if epoch % thp["valid_epoch"] == 0:
        valid_loss = valid_loop(valid_dataloader, model, losses, device)
        writer.add_scalar('loss/valid', valid_loss, epoch)
        if thp['save_best'] and valid_loss < best_score:
            model_save(model_data, epoch, checkpt_path, best=True)
            best_score = valid_loss

    # next step for scheduler
    if thp['scheduler']:
        scheduler.step()
        for n, group in enumerate(optimizer.param_groups):
            logging.info(f'lr for param group {n} is: {group["lr"]}')

    # save
    if epoch % thp["save_epoch"] == 0:
        model_save(model_data, epoch, checkpt_path)
