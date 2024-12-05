import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
from datacollector.normalization import MyoHandSeqNorm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_size, hidden_sizes[i]))
            layers.append(nn.ReLU())
            input_size = hidden_sizes[i]
        layers.append(nn.Linear(input_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])

class HierarchicalModel(nn.Module):
    def __init__(self, sizes, hidden_sizes, path, freeze):
        super(HierarchicalModel, self).__init__()
        self.hand_size = sizes['hand_state']
        self.obj_size = sizes['obj_state']
        self.act_size = sizes['action']
        self.path = path
        self.hand_model = MLP(self.hand_size+self.act_size, hidden_sizes['hand_model'], self.hand_size)
        # self.hand_model.eval()
        self.obj_model = MLP(self.obj_size+self.hand_size, hidden_sizes['external_model'], self.obj_size+self.hand_size)
        # self.obj_model = MLP(self.obj_size+256, hidden_sizes['external_model'], self.obj_size+self.hand_size)
        if path is not None:
            self.load_hand_model(path)
            if freeze:
                for name, param in self.hand_model.named_parameters():
                    param.requires_grad = False

    def forward(self, x ,data_processor):
        hand_state = x[..., :self.hand_size]
        obj_state = x[..., self.hand_size:self.hand_size+self.obj_size]
        action = x[..., self.hand_size+self.obj_size:]
        hand_state = hand_state * torch.from_numpy(data_processor.hand_data_std).to(hand_state.device) + torch.from_numpy(data_processor.hand_data_mean).to(hand_state.device)
        hand_state = (hand_state - MyoHandSeqNorm.mean.to(hand_state.device)) / MyoHandSeqNorm.std.to(hand_state.device)
        # hand_feats = self.hand_model.model[0](torch.cat([hand_state, action], dim=-1).float())
        # hand_feats = self.hand_model.model[1](hand_feats)
        pred_hand_state = self.hand_model(torch.cat([hand_state, action], dim=-1).float())
        pred_hand_state = pred_hand_state * MyoHandSeqNorm.std.to(hand_state.device)+ MyoHandSeqNorm.mean.to(hand_state.device)
        pred_hand_state = (pred_hand_state - torch.from_numpy(data_processor.hand_data_mean).to(hand_state.device)) / torch.from_numpy(data_processor.hand_data_std).to(hand_state.device)
        pred_hand_obj_state = self.obj_model(torch.cat([pred_hand_state, obj_state], dim=-1).float())
        # pred_hand_obj_state = self.obj_model(torch.cat([hand_feats, obj_state], dim=-1).float())
        return pred_hand_obj_state

    def load_hand_model(self, path):
        self.hand_model.load_state_dict(torch.load(path))
    