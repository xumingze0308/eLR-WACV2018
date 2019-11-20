from configs import parse_base_args

__all__ = ['parse_hmdb51_args']

def parse_hmdb51_args():
    parser = parse_base_args()
    parser.add_argument('--test_intervals', default=[40, 45, 50], type=list)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--input_size', default=4096, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--num_classes', default=51, type=int)
    return parser.parse_args()
