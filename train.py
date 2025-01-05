import sys
import logging
import torch
import pathlib
import yaml
import numpy
import random
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from utils.utils import train_loop, valid_loop, model_save, model_load
from data.RefDataset import RefDataset
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


# creating the dataset
train_data = RefDataset(pth['train'], augment=True)
valid_data = RefDataset(pth['valid'], augment=False)

# creating the dataloaders
train_dataloader = DataLoader(train_data, batch_size=thp['batch_size'], shuffle=True, 
                              worker_init_fn=seed_worker, generator=g)
valid_dataloader = DataLoader(valid_data, batch_size=thp['batch_size'], shuffle=True,
                              worker_init_fn=seed_worker, generator=g)

# creating the model
if config["model_name"] == 'FSRST':
    from models.FSRST import FSRST as Net
elif config["model_name"] == 'STAlign128':
    from models.STAlign import STAlign128 as Net
elif config["model_name"] == 'STAlign32':
    from models.STAlign import STAlign32 as Net
else:
    raise ValueError('Invalid model name selected.')

# create the model
model = nn.DataParallel(Net(**mdl)).to(device)
logging.info(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f'Total params: {total_params:,}')


# creating a parameter group for the alignment module
# to give it a lower learning rate
if config["model_name"] == 'FSRST':
    base_params = [p for n, p in model.named_parameters() if not n.startswith('module.feature_align') and p.requires_grad]
    logging.info(f'Number of base params: {len(base_params)}')
    offset_params = [p for n, p in model.named_parameters() if n.startswith('module.feature_align') and p.requires_grad]
    logging.info(f'Number of offset params: {len(offset_params)}')


# create optimizer
if config["model_name"] == 'FSRST':
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
    if config["model_name"] == 'FSRST':
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

