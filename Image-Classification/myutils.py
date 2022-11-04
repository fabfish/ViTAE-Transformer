import os
import sys
import time
import torch
from copy import deepcopy

import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
import logging
import os
from collections import OrderedDict
import math
import torch.nn.functional as F

import re
from einops import rearrange

_logger = logging.getLogger(__name__)

def monarch_weight_to_dense_weight(w1_bfly, w2_bfly):
    """
    Argumments:
        w1_bfly: (nblocks, out / nblocks, in / blocks)
        w2_bfly: (nblocks, out / nblocks, in / blocks)
    Return:
        dense_weight: (out / in)
    """
    # batch, n = x.shape
    device = w1_bfly.device

    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
    w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))

    P_1 = rearrange(torch.eye(k*q), 'b (r l) -> b (l r)', l=l).to(device)
    P_2_T = rearrange(torch.eye(l*s), 'b (l s) -> b (s l)', l=4).to(device)
    P_1_T = P_1.T 
    P_2 = P_2_T.T 

    L = w2_dense
    R = w1_dense

    M = P_2 @ L @ P_1_T @ R 
    return M

def monarch_to_dense_mlp_NC(state_dict, new_state_dict):
    # from src.ops.blockdiag_multiply import blockdiag_weight_to_dense_weight
    blkdiag1_names = sorted({name for name in state_dict
                             if re.match('layers.(\d+).NC.(\d+).mlp.fc(1|2).blkdiag1',
                                          name)})
    blkdiag2_names = sorted({name for name in state_dict
                             if re.match('layers.(\d+).NC.(\d+).mlp.fc(1|2).blkdiag2',
                                          name)})
    _logger.info('Processing S2D: Normal Cell Monarch to Mlp')
    # print(blkdiag1_names)
    # print(blkdiag2_names)
    for blkdiag1_name in blkdiag1_names:
        try:
            blkdiag2_name = blkdiag1_name.replace('blkdiag1', 'blkdiag2')
            new_name = blkdiag1_name.replace('blkdiag1', 'weight')
            new_state_dict[new_name] = monarch_weight_to_dense_weight(state_dict[blkdiag1_name], state_dict[blkdiag2_name])
        except:
            _logger.error("blkdiag and mlp name or device do not match")
        # blkdiag2_name = blkdiag1_name.replace('blkdiag1', 'blkdiag2')
        # new_name = blkdiag1_name.replace('blkdiag1', 'weight')
        # new_state_dict[new_name] = monarch_weight_to_dense_weight(state_dict[blkdiag1_name], state_dict[blkdiag2_name])
    return new_state_dict