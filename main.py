import os
import torch
from lib.trainer import Trainer


class Args:

    def __init__(self):
        self.data_dir = './small_parallel_enja'
        self.vocab_size = 75000
        self.batch_size = 32
        self.encoder_hidden_n = 512
        self.encoder_layers_n = 3
        self.encoder_embedding_dim = 300
        self.encoder_bidirec = True
        self.decoder_hidden_n = 512
        self.decoder_layers_n = 3
        self.decoder_embedding_dim = 512
        self.dropout_p = 0.1
        self.use_cuda = True
        self.gpu_id = 2


if __name__ == '__main__':
    args = Args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    trainer = Trainer(args)
    trainer.train_one_epoch(1)
