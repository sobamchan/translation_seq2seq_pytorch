import torch
from torch.autograd import Variable
from lib import dataset
from lib import utils
from lib import models


class Trainer:

    def __init__(self, args):
        self.args = args
        train_dataloader, test_dataloader = dataset.get_dataloader(args)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.src_vocab = train_dataloader.dataset.src_vocab
        self.tgt_vocab = train_dataloader.dataset.tgt_vocab

        # set model
        encoder = models.Encoder(self.src_vocab, args)
        if args.use_cuda:
            self.encoder = encoder.cuda()
        else:
            self.encoder = encoder
        # set optimizer
        # set criteria

    def train_one_epoch(self, i_epoch):
        args = self.args
        self.encoder.train()
        self.encoder.zero_grad()
        # losses = []

        for i, dict_ in enumerate(self.train_dataloader):
            src_sents = dict_['src']
            src_sents = [self.src_vocab.encode(src_sent)[0]
                         for src_sent in src_sents]
            tgt_sents = dict_['tgt']
            tgt_sents = [self.tgt_vocab.encode(tgt_sent)[0]
                         for tgt_sent in tgt_sents]

            src_sents, src_lens, tgt_sents, tgt_lens =\
                utils.pad_to_batch(src_sents,
                                   tgt_sents,
                                   self.src_vocab.w2i['<PAD>'],
                                   self.tgt_vocab.w2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents))
            tgt_sents = Variable(torch.LongTensor(tgt_sents))
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            self.encoder(src_sents, src_lens)
