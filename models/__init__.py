import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F

from .dataset.dataset import PoseDataset, PositionDataset, ActionDataset, TransitionDataset, InverseTransitionDataset, IsaacInverseTransitionDataset

import os
from torch.utils.data import DataLoader
import time
from torch import optim
from fastprogress import progress_bar
import wandb
import logging
from torch.utils.tensorboard import SummaryWriter

from .mlp import MLP, HierarchicalModel

class Trainer:
    def __init__(self, args):
        if args.pred_type == 'inverse':
            self.net = MLP(args.input_size, args.hidden_sizes, args.output_size).to(args.device)
        else:
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
        
        if self.args.pred_type == 'position' or self.args.pred_type == 'position_seq':
            self.loss_func = nn.MSELoss()
        elif self.args.pred_type == 'inverse' or self.args.pred_type == 'inverse_seq':
            self.loss_func = nn.L1Loss()

        self.writer = SummaryWriter(log_dir=self.log_path)

        self.train_loss_step = 0

        self.val_loss_step = 0
        
        
    def get_data(self, args):
        if args.pred_type == 'position':
            self.train_set = PositionDataset(data_path=args.train_path, normalize=args.normalization, device=self.device)
            self.val_set = PositionDataset(data_path=args.val_path, normalize=args.normalization, device=self.device)
        elif args.pred_type == 'pose':
            self.train_set = PoseDataset(data_path=args.train_path, normalize=args.normalization, device=self.device)
            self.val_set = PoseDataset(data_path=args.val_path, normalize=args.normalization, device=self.device)
        elif args.pred_type == 'inverse':
            self.train_set = ActionDataset(data_path=args.train_path, normalize=args.normalization, device=self.device)
            self.val_set = ActionDataset(data_path=args.val_path, normalize=args.normalization, device=self.device)
        elif args.pred_type == 'position_seq':
            self.train_set = TransitionDataset(data_path=args.train_path, device=self.device)
            self.val_set = TransitionDataset(data_path=args.val_path, device=self.device)
        elif args.pred_type == 'inverse_seq':
            # self.train_set = InverseTransitionDataset(data_path=args.train_path, device=self.device)
            # self.val_set = InverseTransitionDataset(data_path=args.val_path, device=self.device)
            self.train_set = IsaacInverseTransitionDataset(data_path=args.train_path, device=self.device)
            self.val_set = IsaacInverseTransitionDataset(data_path=args.val_path, device=self.device)
        else: 
            AssertionError(f"{args.pred_type} dataset not implemented !!!")
            
        train_dataloader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(self.val_set, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
        return train_dataloader, val_dataloader
    
    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        

    def one_epoch(self, train=True,):
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
                data = data.to(self.device)
                labels = labels.to(self.device)
                preds = self.net(data)
                loss = self.loss_func(labels, preds)
                avg_loss += loss
            if train:
                self.train_step(loss)
                # wandb.log({"train_mse": loss.item(),
                #             "learning_rate": self.optimizer.state_dict()['param_groups'][0]['lr']})
                self.writer.add_scalar("train_mse", loss.item(), self.train_loss_step)
                self.train_loss_step +=1
                pbar.comment = f"Loss={loss.item():.10f}"
            else:
                if self.args.pred_type == 'position' or self.args.pred_type == 'position_seq':
                    if self.args.normalization:
                        unnormalized_preds = preds * self.train_set.std.to(self.device) + self.train_set.mean.to(self.device)
                        unnormalized_labels = labels * self.train_set.std.to(self.device) + self.train_set.mean.to(self.device)
                    else:
                        unnormalized_preds = preds
                        unnormalized_labels = labels
                    dist_error = torch.norm(unnormalized_preds - unnormalized_labels, dim=-1).mean()
                    # dist_error = torch.norm(preds - labels, dim=-1).mean()
                    pbar.comment = f"Evaluating...Dist Error={dist_error:.10f}"
                    error += dist_error
                elif self.args.pred_type == 'inverse' or self.args.pred_type == 'inverse_seq':
                    action_error = torch.abs(preds - labels).mean()
                    pbar.comment = f"Evaluating...Action Error={action_error:.10f}"
                    error += action_error
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



def weighted_mse_loss(preds, labels, alpha=.98):
    """
        preds: batch_size x seq_len x output_size
        labels: batch_size x seq_len x output_size
    """
    weights = alpha ** torch.arange(labels.shape[1], device=labels.device)
    losses = torch.sum(weights.unsqueeze(0) * torch.norm(preds - labels, dim=-1) ** 2, dim=-1) / torch.sum(weights)
    return losses.mean()

class NNDynModel:
    def __init__(self, args, data_processor):
        if args.use_handmodel:
            self.net = HierarchicalModel(args.sizes, args.hidden_sizes, args.hand_model_path, args.freeze_handmodel).to(args.device)
        else:
            self.net = MLP(args.input_size, args.hidden_sizes, args.output_size).to(args.device)
        self.device = args.device
        self.args = args
        self.data_processor = data_processor

    def prepare(self, args):
        # prepare log
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        run_name = args.exp_name + cur_time
        self.log_path = os.path.join(args.log_path, f'{run_name}')
        os.makedirs(self.log_path, exist_ok=True)
        
        if args.optimizer == "Adam":
            if args.freeze_handmodel:
                self.optimizer = torch.optim.Adam(self.net.obj_model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay) 
            else:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            if args.freeze_handmodel:
                self.optimizer = torch.optim.SGD(self.net.obj_model.parameters(), lr=args.lr, momentum=0.9)
            else:
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        else:
            raise AssertionError(f"Unknown optimizer: {args.optimizer}")
        
        # if args.scheduler == 'Linear':
        #     self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        # elif args.scheduler == 'Adaptive':
        #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=args.factor, threshold=args.threshold, patience=args.patience, verbose=True)
        # else:
        #     raise AssertionError(f"Unknown scheduler: {args.scheduler}")
        # self.scheduler_name = args.scheduler
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.loss_func = weighted_mse_loss

        self.writer = SummaryWriter(log_dir=self.log_path)

        self.train_loss_step = 0

        self.val_loss_step = 0
        
        
    def get_data(self, train_set, val_set, args):
        self.train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        
    
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
        step = 0
        for i, (data, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.no_grad() if not train else torch.enable_grad()):
                init_state = torch.concat([data[0], data[1]], dim=-1).float().to(self.device)
                actions = data[2].float().to(self.device)
                next_states = torch.concat([labels[0], labels[1]], dim=-1).float().to(self.device)
                
                preds = []
                state = init_state
                for t in range(next_states.shape[1]):
                    if self.args.use_handmodel:
                        pred = self.net(torch.concat([state, actions[:, t]], dim=-1), self.data_processor)
                    else:
                        pred = self.net(torch.concat([state, actions[:, t]], dim=-1))
                    state = pred
                    preds.append(pred)
                preds = torch.stack(preds, dim=1)   # batch_size x seq_len x output_size
                loss = self.loss_func(preds, next_states)
                avg_loss += loss.detach().cpu().item()
            if train:
                self.train_step(loss)
                self.writer.add_scalar("weighted_train_mse", loss.item(), self.train_loss_step)
                self.train_loss_step +=1
                pbar.comment = f"Loss={loss.item():.10f}"
            else:
                dist_error = torch.norm(preds - next_states, dim=-1).mean().detach().cpu().item()

                error += dist_error
            step += 1
        return avg_loss/(step+1), error/(step+1)
    
    def fit(self, epochs=30, do_validation=True):
        losses = 0
        errors = 0
        for epoch in progress_bar(range(epochs)):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            if do_validation:
                avg_loss, val_loss = self.one_epoch(train=False,)
                # if self.scheduler_name == 'Adaptive':
                #     self.scheduler.step(avg_loss)
                # else: 
                #     self.scheduler.step()
                self.writer.add_scalar("val_loss", avg_loss, self.val_loss_step)
                self.writer.add_scalar("val_error", val_loss, self.val_loss_step)
                self.val_loss_step += 1
                losses += avg_loss
                errors += val_loss
        return losses/epochs, errors/epochs
                
    def save_model(self, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.net.state_dict(), os.path.join(self.log_path, f"ckpt{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_path, f"optim{epoch}.pt"))
        
    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))

    def predict(self, cur_states, actions):
        batch, seq_len, output_size = actions.shape
        preds = []
        state = cur_states
        for t in range(seq_len):
            if self.args.use_handmodel:
                pred = self.net(torch.concat([state, actions[:,t]], dim=-1), self.data_processor)
            else:
                pred = self.net(torch.concat([state, actions[:,t]], dim=-1))
            state = pred
            preds.append(pred)
        preds = torch.stack(preds, dim=1) # batch x seq_len x output_size
        return preds