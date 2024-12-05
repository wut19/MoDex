import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F

from models.dataset.dataset import SeqTransDataset, IsaacSeqTransDataset

import os
from torch.utils.data import DataLoader
import time
from torch import optim
from fastprogress import progress_bar
import logging
from torch.utils.tensorboard import SummaryWriter

from models.mlp import MLP

from omegaconf import OmegaConf
import argparse


def weighted_mse_loss(preds, labels, alpha=.95):
    """
        preds: batch_size x seq_len x output_size
        labels: batch_size x seq_len x output_size
    """
    weights = alpha ** torch.arange(labels.shape[1], device=labels.device)
    losses = torch.sum(weights.unsqueeze(0) * torch.norm(preds - labels, dim=-1) ** 2, dim=-1) / torch.sum(weights)
    return losses.mean()

class SeqTrainer:
    def __init__(self, args):
        self.net = MLP(args.input_size, args.hidden_sizes, args.output_size).to(args.device)
        self.device = args.device
        self.args = args

    def prepare(self, args):
        # prepare log
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        run_name = args.exp_name + cur_time
        self.log_path = os.path.join(args.log_path, f'{run_name}')
        os.makedirs(self.log_path, exist_ok=True)
        
        self.train_dataloader, self.val_dataloader = self.get_data(args)
        
        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        else:
            raise AssertionError(f"Unknown optimizer: {args.optimizer}")
        
        if args.scheduler == 'Linear':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'Adaptive':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=args.factor, threshold=args.threshold, patience=args.patience, verbose=True)
        else:
            raise AssertionError(f"Unknown scheduler: {args.scheduler}")
        self.scheduler_name = args.scheduler
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.loss_func = weighted_mse_loss

        self.writer = SummaryWriter(log_dir=self.log_path)

        self.train_loss_step = 0

        self.val_loss_step = 0
        
        
    def get_data(self, args):
        if args.normalization:
            self.train_set = SeqTransDataset(data_path=args.train_path, device=self.device)
            self.val_set = SeqTransDataset(data_path=args.val_path, device=self.device)
        else:
            self.train_set = IsaacSeqTransDataset(data_path=args.train_path, device=self.device)
            self.val_set = IsaacSeqTransDataset(data_path=args.val_path, device=self.device)

        train_dataloader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(self.val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
        return train_dataloader, val_dataloader
    
    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        

    def one_epoch(self, train=True):
        avg_loss = 0
        error = 0
        if train:
            self.net.train()
            pbar = progress_bar(self.train_dataloader, leave=False)
        else:
            self.net.eval()
            pbar = progress_bar(self.val_dataloader, leave=False)
        for i, (data, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.no_grad() if not train else torch.enable_grad()):
                init_state = data[0].to(self.device)
                actions = data[1].to(self.device)
                next_states = labels.to(self.device)
                
                preds = []
                state = init_state
                for t in range(next_states.shape[1]):
                    pred = self.net(torch.concat([state, actions[:, t]], dim=-1))
                    state = pred
                    preds.append(pred)
                preds = torch.stack(preds, dim=1)   # batch_size x seq_len x output_size
                loss = self.loss_func(preds, next_states)
                avg_loss += loss
            if train:
                self.train_step(loss)
                # wandb.log({"train_mse": loss.item(),
                #             "learning_rate": self.optimizer.state_dict()['param_groups'][0]['lr']})
                self.writer.add_scalar("weighted_train_mse", loss.item(), self.train_loss_step)
                self.train_loss_step +=1
                pbar.comment = f"Loss={loss.item():.10f}"
            else:
                if self.args.normalization:
                    unnormalized_preds = preds * self.train_set.std.to(self.device) + self.train_set.mean.to(self.device)
                    unnormalized_next_states = next_states * self.train_set.std.to(self.device) + self.train_set.mean.to(self.device)
                    dist_error = torch.norm(unnormalized_preds - unnormalized_next_states, dim=-1).mean()
                else:
                    dist_error = torch.norm(preds - next_states, dim=-1).mean()
                pbar.comment = f"Evaluating...Dist Error={dist_error:.10f}"
                error += dist_error
        return avg_loss/(i+1), error/(i+1)
    
    def fit(self, args):
        for epoch in progress_bar(range(args.epochs)):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            if args.do_validation:
                avg_loss, error = self.one_epoch(train=False,)
                if self.scheduler_name == 'Adaptive':
                    self.scheduler.step(avg_loss)
                else: 
                    self.scheduler.step()
                # wandb.log({"val_mse": avg_loss})
                self.writer.add_scalar("val_mse", avg_loss, self.val_loss_step)
                self.writer.add_scalar("error", error, self.val_loss_step)
                self.val_loss_step += 1
                print('val_mse:', avg_loss)
                print('error:', error)

            if epoch % args.save_interval == 0:
                self.save_model(epoch)
        self.save_model()

                
    def save_model(self, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.net.state_dict(), os.path.join(self.log_path, f"ckpt{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_path, f"optim{epoch}.pt"))
        # at = wandb.Artifact("model", type="model", description="Model weights", metadata={"epoch": epoch})
        # at.add_dir(self.log_path)
        # wandb.log_artifact(at)
        
    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))


def train(args):
    cfg = OmegaConf.load(args.cfg)
    model = SeqTrainer(cfg)
    if args.load_model_path:
        model.net.load_state_dict(torch.load(args.load_model_path))
    cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
    # with wandb.init(project=cfg.exp_name, name=cur_time, group='train', config=cfg):
    model.prepare(cfg)
    model.fit(cfg)
    model.writer.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='path of training configuration file')
    parser.add_argument('--load_model_path', type=str, default=None, help='trained model path' )
    args = parser.parse_args()
    train(args)