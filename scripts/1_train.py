import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models import Trainer
from omegaconf import OmegaConf
import wandb
import time
import argparse
import torch

def train(args):
    cfg = OmegaConf.load(args.cfg)
    model = Trainer(cfg)
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