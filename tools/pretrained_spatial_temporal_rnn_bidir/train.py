import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
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

    datasets = {
        phase: DataLayer(
            data_root=osp.join(args.data_root, phase),
            phase=phase,
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: data.DataLoader(
            datasets[phase],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for phase in args.phases
    }

    model = Model(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        bidirectional=args.bidirectional,
        num_classes=args.num_classes,
    ).apply(utl.weights_init).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim=1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        losses = {phase: 0.0 for phase in args.phases}
        corrects = {phase: 0.0 for phase in args.phases}

        start = time.time()
        for phase in args.phases:
            training = 'Test' not in phase
            if training:
                model.train(True)
            else:
                if epoch in args.test_intervals:
                    model.train(False)
                else:
                    continue

            with torch.set_grad_enabled(training):
                for batch_idx, (spatial, temporal, length, target) in enumerate(data_loaders[phase]):
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
                    loss = criterion(score, target_input)
                    losses[phase] += loss.item() * target_input.shape[0]
                    if args.debug:
                        print(loss.item())

                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        pred = torch.max(softmax(score), 1)[1].cpu()
                        corrects[phase] += torch.sum(pred == target_input.cpu()).item()
        end = time.time()

        print('Epoch {:2} | '
              'Train loss: {:.5f} Val loss: {:.5f} | '
              'Test loss: {:.5f} accuracy: {:.5f} | '
              'running time: {:.2f} sec'.format(
                  epoch,
                  losses['Train']/len(data_loaders['Train'].dataset),
                  losses['Validation']/len(data_loaders['Validation'].dataset),
                  losses['Test']/len(data_loaders['Test'].dataset),
                  corrects['Test']/len(data_loaders['Test'].dataset),
                  end - start,
              ))

        if epoch in args.test_intervals:
            torch.save(model.state_dict(), osp.join(this_dir, './state_dict-epoch-'+str(epoch)+'.pth'))

if __name__ == '__main__':
    main(parse_args())
