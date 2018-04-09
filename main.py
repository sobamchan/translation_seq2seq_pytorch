import os
import torch
import argparse
from distutils.util import strtobool
from lib.trainer import Trainer


class Args:

    def __init__(self):
        self.data_dir = './small_parallel_enja'
        self.epoch = 100
        self.src_vocab_size = 25000
        self.tgt_vocab_size = 25000
        self.batch_size = 32
        self.encoder_hidden_n = 512
        self.encoder_layers_n = 3
        self.encoder_embedding_dim = 512
        self.encoder_bidirec = True
        self.decoder_hidden_n = 512 * 2
        self.decoder_layers_n = 3
        self.decoder_embedding_dim = 512
        self.dropout_p = 0.1
        self.lr = 0.0001
        self.use_cuda = True
        self.gpu_id = 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        default='./small_parallel_enja')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--src-vocab-size', type=int, default=25000)
    parser.add_argument('--tgt-vocab-size', type=int, default=25000)
    parser.add_argument('--encoder-hidden-n', type=int, default=512)
    parser.add_argument('--encoder-layers-n', type=int, default=3)
    parser.add_argument('--encoder-embedding-dim', type=int, default=512)
    parser.add_argument('--encoder-bidirec', type=strtobool, default='1')
    parser.add_argument('--decoder-hidden-n', type=int, default=512)
    parser.add_argument('--decoder-layers-n', type=int, default=3)
    parser.add_argument('--decoder-embedding-dim', type=int, default=512)
    parser.add_argument('--dropout-p', type=float, default=0.1)
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--gpu-id', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    # args = Args()
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    trainer = Trainer(args)

    for i_epoch in range(1, args.epoch + 1):
        loss = trainer.train_one_epoch(i_epoch)
        print('%d th epoch: loss -> %f' % (i_epoch, loss))
