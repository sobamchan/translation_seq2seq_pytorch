import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
        encoder = models.Encoder(self.src_vocab, self.args)
        encoder.init_weight()
        decoder = models.Decoder(self.tgt_vocab, self.args)
        decoder.init_weight()
        if args.use_cuda:
            self.encoder = encoder.cuda()
            self.decoder = decoder.cuda()
        else:
            self.encoder = encoder
            self.decoder = decoder
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=args.lr)
        self.criteria = nn.CrossEntropyLoss(ignore_index=0)

    def train_one_epoch(self, i_epoch):
        args = self.args
        self.encoder.train()
        self.decoder.train()
        losses = []

        total = int(len(self.train_dataloader.dataset) / args.batch_size)
        for i, dict_ in tqdm(enumerate(self.train_dataloader), total=total):
            self.encoder.zero_grad()
            self.decoder.zero_grad()
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
            batch_size, tgt_len = tgt_sents.size()
            start_decode =\
                Variable(torch.LongTensor([[self.tgt_vocab.w2i['<s>']] *
                                          batch_size])).transpose(0, 1)
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(src_sents, src_lens)
            preds = self.decoder(start_decode,
                                 hidden_c,
                                 tgt_len,
                                 output)
            loss = self.criteria(preds, tgt_sents.view(-1))
            loss.backward()
            losses.append(loss.data[0])
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 50.0)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 50.0)
            self.encoder_optim.step()
            self.decoder_optim.step()

        return np.mean(losses)

    def test(self):
        args = self.args
        for i, dict_ in enumerate(self.test_dataloader):
            src_sents = dict_['src']
            print('source sentence: %s' % src_sents[0])
            tgt_sents = dict_['tgt']
            print('target sentence: %s' % tgt_sents[0])

            src_sents = [self.src_vocab.encode(src_sent)[0]
                         for src_sent in src_sents]
            tgt_sents = [self.tgt_vocab.encode(tgt_sent)[0]
                         for tgt_sent in tgt_sents]

            src_sents, src_lens, tgt_sents, tgt_lens =\
                utils.pad_to_batch(src_sents,
                                   tgt_sents,
                                   self.src_vocab.w2i['<PAD>'],
                                   self.tgt_vocab.w2i['<PAD>'])
            src_sents = Variable(torch.LongTensor(src_sents))
            tgt_sents = Variable(torch.LongTensor(tgt_sents))
            batch_size, tgt_len = tgt_sents.size()
            start_decode =\
                Variable(torch.LongTensor([[self.tgt_vocab.w2i['<s>']] *
                                          batch_size])).transpose(0, 1)
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(src_sents, src_lens)
            preds = self.decoder(start_decode,
                                 hidden_c,
                                 tgt_len,
                                 output)

            _, preds_max = torch.max(preds.view(tgt_sents.size(0),
                                                tgt_len,
                                                len(self.tgt_vocab)), 2)
            print('predicted sentence: %s' %
                  self.tgt_vocab.decode(preds_max[0].data))
            break
