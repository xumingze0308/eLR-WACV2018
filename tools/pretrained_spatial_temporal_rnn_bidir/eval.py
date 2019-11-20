import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence

import _init_paths
import utils as utl
from configs.hmdb51 import parse_hmdb51_args as parse_args
from datasets import HMDB51DataLayer as DataLayer
from models import RNN as Model

def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = data.DataLoader(
        DataLayer(
            data_root=osp.join(args.data_root, 'Test'),
            phase='Test',
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = Model().to(device)
    model.load_state_dict(checkpoint)
    model.train(False)
    softmax = nn.Softmax(dim=1).to(device)

    corrects = 0.0
    with torch.set_grad_enabled(False):
        for batch_idx, (spatial, temporal, length, target) in enumerate(data_loader):
            spatial_input = torch.zeros(*spatial.shape)
            temporal_input = torch.zeros(*temporal.shape)
            target_input = []
            length_input = []

            index = utl.argsort(length)[::-1]
            for i, idx in enumerate(index):
                spatial_input[i] = spatial[idx]
                temporal_input[i] = temporal[idx]
                target_input.append(target[idx])
                length_input.append(length[idx])

            spatial_input = spatial_input.to(device)
            temporal_input = temporal_input.to(device)
            target_input = torch.LongTensor(target_input).to(device)
            pack1 = pack_padded_sequence(spatial_input, length_input, batch_first=True)
            pack2 = pack_padded_sequence(temporal_input, length_input, batch_first=True)

            score = model(pack1, pack2)
            pred = torch.max(softmax(score), 1)[1].cpu()
            corrects += torch.sum(pred == target_input.cpu()).item()

    print('The accuracy is {:.4f}'.format(corrects/len(data_loader.dataset)))

if __name__ == '__main__':
    main(parse_args())

