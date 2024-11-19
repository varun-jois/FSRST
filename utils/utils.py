
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from models.Transformer import RSRT as Net
from torchvision.utils import save_image
import torch.nn as nn
import logging

#####################################################
# main train and validation loops
def train_loop(dataloader, model_data, losses, device):
    """
    The main training loop
    """
    model, optimizer = model_data['mdl'], model_data['opt']
    epoch_loss = 0
    model.train()
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        # load onto device
        hq, lq, refs = data
        hq, lq = hq.to(device), lq.to(device)
        if refs:
            for i in range(len(refs)):
                refs[i] = refs[i].to(device)
            pred = model(lq, refs)
        else:
            pred = model(lq)
        
        # Compute prediction loss
        loss = losses['loss_rec'](pred, hq)
        # Backpropagation
        loss.backward()
        # params = {n: p.max().item() for n, p in model.named_parameters()}
        # p_max = max(params, key=lambda x: params.get(x))
        # p_min = min(params, key=lambda x: params.get(x))
        # logging.info(f'param max {p_max} -> {params[p_max]}')
        # logging.info(f'param min {p_min} -> {params[p_min]}')
        # logging.info(f'pred max min {pred.max().item()} | {pred.min().item()}')
        # logging.info(f'loss {loss.item()}')
        if loss.isinf().any().item():
            logging.info('Saving in checkpoints/anomaly')
            torch.save(model.state_dict(), f'/home/varun/fsrt/checkpoints/anomaly/mdl.pth')
            raise 
        # grads = [p.grad.detach().flatten() for p in model.parameters() 
        #          if p.grad is not None]
        # norm = torch.cat(grads)
        # vals = [norm.min().item(), norm.mean().item(), norm.max().item(), norm.norm().item()]
        # logging.info(f'mean before clip: {vals}')
        # norm = clip_grad_norm_(model.parameters(), 0.9)
        # grads = [p.grad.detach().flatten() for p in model.parameters() 
        #          if p.grad is not None]
        # norm = torch.cat(grads)
        # vals = [norm.min().item(), norm.mean().item(), norm.max().item(), norm.norm().item()]
        # logging.info(f'norm after clip: {vals}')
        optimizer.step()

        # save the epoch loss
        epoch_loss += loss.item()
    epoch_loss /= (batch + 1)
    #print(f'Train loss: {epoch_loss:.5f}')
    logging.info(f'Train loss: {epoch_loss:.5f}')
    return epoch_loss
    


def valid_loop(dataloader, model, losses, device):
    """
    The main validation loop
    """
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # load onto device
            hq, lq, refs = data
            hq, lq = hq.to(device), lq.to(device)
            if refs:
                for i in range(len(refs)):
                    refs[i] = refs[i].to(device)
                pred = model(lq, refs)
            else:
                pred = model(lq)
            # Compute loss
            loss = losses['loss_rec'](pred, hq)
            epoch_loss += loss.item()   
        epoch_loss /= (batch + 1)
        #print(f'Valid loss: {epoch_loss:.5f}')
        logging.info(f'Valid loss: {epoch_loss:.5f}')
    return epoch_loss


#####################################################
# model saving and loading
def model_save(model_data, epoch, checkpt_path, best=False):
    for name, model in model_data.items():
         #print(f'Saving {checkpt_path}/{epoch}_{name}.pth')
         if best:
             logging.info(f'Saving {checkpt_path}/best_{name}.pth')
             torch.save(model.state_dict(), f'{checkpt_path}/best_{name}.pth')
         else:
            logging.info(f'Saving {checkpt_path}/{epoch}_{name}.pth')
            torch.save(model.state_dict(), f'{checkpt_path}/{epoch}_{name}.pth')


def model_load(model_data, epoch_start, checkpt_path):
    for name, model in model_data.items():
        #print(f'Loading weights {checkpt_path}/{epoch_start}_{name}.pth')
        logging.info(f'Loading weights {checkpt_path}/{epoch_start}_{name}.pth')
        model.load_state_dict(torch.load(f'{checkpt_path}/{epoch_start}_{name}.pth'))
    


#####################################################
# extras

def performance_loop(dataloader, model, losses, device, metrics, img_path=None):
    """
    The main validation loop
    """
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # load onto device
            hq, lq, refs = data
            hq, lq = hq.to(device), lq.to(device)
            for i in range(len(refs)):
                refs[i] = refs[i].to(device)
            
            # Compute prediction and loss
            pred = model(lq, refs)
            print(f'Pred range: {pred.max() - pred.min()}')
            print(f'hq range: {hq.max() - hq.min()}')
            loss = losses['loss_rec'](pred, hq)
            valid_loss += loss.item()

            # save two sets of images
            if img_path is not None and batch == 0:
                output_size = hq.shape[2]
                bic = F.interpolate(lq, (output_size, output_size), mode='bicubic', align_corners=False)
                for i in range(2):
                    save_image(bic[i], f"{img_path}/{i}_bi.png")
                    save_image(pred[i], f"{img_path}/{i}_sr.png")
                    save_image(hq[i], f"{img_path}/{i}_hq.png")
                    save_image(lq[i], f"{img_path}/{i}_lq.png")

            # unnormalize data to [0, 255] range. Taken from save_image in torchvision
            pred = pred.mul(255).add_(0.5).clamp_(0, 255).trunc()
            hq = hq.mul(255).add_(0.5).clamp_(0, 255).trunc()

            # compute metrics
            if metrics is not None:
                for m in metrics:
                    if m == 'lpips':  # need to bring to 0 to 1 range
                        _ = metrics[m](pred.div(255), hq.div(255))
                    else:
                        _ = metrics[m](pred, hq)
            
    print(f'loss: {valid_loss / (batch + 1)}')
    if metrics is not None:
        # final metric computation
        for m in metrics:
            score = metrics[m].compute()
            print(f'{m}: {score:.4f}')
