import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import init


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


'''
the distribution layer which apply specific layer for each dst
'''
class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node):
        super(MultiCategorical, self).__init__()
        
        linears = []
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        for i in range(num_node):
            linear = init_(nn.Linear(num_inputs, num_outputs))
            linears.append(linear)
        self.linears = nn.ModuleList(linears)

    '''
    @param:
        x: shape[(len, )batch, num_inputs]
        dst_state: shape[(len, )batch, num_node], num_node = num outputs
    @retval:
        a distribution: shape[(len, )batch, num_outputs]
    '''
    def forward(self, x, dst_state):
        xs = []
        for linear in self.linears:
            xs.append(linear(x))
        concat_x = torch.stack(xs, -2) 
        result = concat_x * dst_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        
        return torch.distributions.Categorical(logits=result)


'''
the distribution layer which apply specific layer for each dst and each type 
layer num = m(type) + n(dst)
'''
class MultiTypeCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node, num_type):
        super(MultiTypeCategorical, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        
        dst_type_linears = []
        for i in range(num_node * num_type):
            linear = init_(nn.Linear(num_inputs, num_outputs)) 
            dst_type_linears.append(linear)
        self.dst_type_linears = nn.ModuleList(dst_type_linears)
    
    '''
    @param:
        x: shape[(len,)batch, num_inputs]
        dst_state: shape[(len,)batch, num_node]
    @retval:
        a distribution: shape[(len, )batch, num_outputs]
    '''
    def forward(self, x, dst_state, type_state): 
        xs = []
        for linear in self.dst_type_linears:
            xs.append(linear(x))
        concat_x = torch.stack(xs, -2) 
        dst_type_state = torch.matmul(dst_state.unsqueeze(-1), type_state.unsqueeze(-2))
        state_shape = list(dst_type_state.size())
        if len(state_shape) == 4:
            dst_type_state = dst_type_state.view(state_shape[0], state_shape[1], -1)
        elif len(state_shape) == 3:
            dst_type_state = dst_type_state.view(state_shape[0], -1)
        else:
            raise NotImplementedError
        result = concat_x * dst_type_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(logits=result)
        
