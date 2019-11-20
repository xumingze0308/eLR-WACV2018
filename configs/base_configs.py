import argparse

__all__ = ['parse_base_args']

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/HMDB51/Split1', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--phases', default=['Validation', 'Train', 'Test'], type=list)
    return parser

