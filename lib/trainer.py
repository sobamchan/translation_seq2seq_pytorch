import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor
from tqdm import tqdm
from lib.dataset import get_dataloader
from lib import utils
from lib import models


class Trainer:

    def __init__(self, args):
        self.args = args
        self.logger = args.logger

        if args.seed > 1:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)

        self.use_cuda = args.use_cuda
        train_loader, test_loader = get_dataloader(args)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

        self.src_vocab = train_loader.dataset.src_vocab
        self.tgt_vocab = train_loader.dataset.tgt_vocab

        self.set_models()

    def set_models(self):
        args = self.args
        encoder = models.Encoder(self.train_loader.dataset.src_vocab, args)
        decoder = models.Decoder(self.train_loader.dataset.tgt_vocab, args)
        encoder.init_weight()
        decoder.init_weight()
        if self.use_cuda:
            self.encoder = encoder.cuda()
            self.decoder = decoder.cuda()
        else:
            self.encoder = encoder
            self.decoder = decoder
        self.enc_optim = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.dec_optim = optim.Adam(self.decoder.parameters(), lr=args.lr)

    def train_one_epoch(self, log_dict):
        tgt_vocab = self.tgt_vocab
        args = self.args
        total = int(len(self.train_loader.dataset) / args.batch_size)
        for i, dict_ in tqdm(enumerate(self.train_loader), total=total):
            src_sents, src_lens, tgt_sents, tgt_lens =\
                self.prepare_batch(dict_['src'], dict_['tgt'])

            batch_size = src_sents.size(0)
            y_len = max(tgt_lens)

            start_decode =\
                Variable(LongTensor([[tgt_vocab.w2i['<s>']] *
                         batch_size])).transpose(0, 1)

            if self.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()
                start_decode = start_decode.cuda()

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            output, hidden_c = self.encoder(src_sents, src_lens)
            preds = self.decoder(start_decode, hidden_c,
                                 y_len, output)

            # MSE loss
            loss = self.loss_func(preds, tgt_sents.view(-1))
            loss.backward()
            log_dict['train_losses'].append(loss.data[0])
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 50.0)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 50.0)
            self.enc_optim.step()
            self.dec_optim.step()

    def prepare_batch(self, src_sents, tgt_sents):
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
        src_sents = [src_vocab.encode(src_sent)[0]
                     for src_sent in src_sents]
        tgt_sents = [tgt_vocab.encode(tgt_sent)[0]
                     for tgt_sent in tgt_sents]
        src_sents, src_lens, tgt_sents, tgt_lens =\
            utils.pad_to_batch(src_sents,
                               tgt_sents,
                               src_vocab.w2i['<PAD>'],
                               tgt_vocab.w2i['<PAD>'])
        src_sents = Variable(torch.LongTensor(src_sents))
        tgt_sents = Variable(torch.LongTensor(tgt_sents))
        return src_sents, src_lens, tgt_sents, tgt_lens

    def translate_batch(self, src_sents, tgt_sents):
        src_sents, src_lens, tgt_sents, tgt_lens =\
            self.prepare_batch(src_sents, tgt_sents)
        batch_size = src_sents.size(0)
        start_decode =\
            Variable(LongTensor([[self.tgt_vocab.w2i['<s>']] *
                     batch_size])).transpose(0, 1)
        if self.use_cuda:
            src_sents = src_sents.cuda()
            tgt_sents = tgt_sents.cuda()
            start_decode = start_decode.cuda()
        output, hidden_c = self.encoder(src_sents, src_lens)
        preds, attn = self.decoder.decode(start_decode,
                                          hidden_c,
                                          output)
        pred_sents = []
        for pred_idx in range(batch_size):
            pred = preds.data[pred_idx]
            pred_sent = self.tgt_vocab.decode(pred)
            pred_sents.append(pred_sent)
        return pred_sents

    def translate(self, pair):
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab

        src_sent = pair[0]
        tgt_sent = pair[1]
        src_sent = src_vocab.encode(src_sent)[0]
        tgt_sent = tgt_vocab.encode(tgt_sent)[0]
        src_sent, _, tgt_sent, _ = utils.pad_to_batch([src_sent],
                                                      [tgt_sent],
                                                      src_vocab.w2i['<PAD>'],
                                                      tgt_vocab.w2i['<PAD>'])
        src_sent = Variable(torch.LongTensor(src_sent))
        tgt_sent = Variable(torch.LongTensor(tgt_sent))
        if self.use_cuda:
            src_sent = src_sent.cuda()
            tgt_sent = tgt_sent.cuda()
        output, hidden_c = self.encoder(src_sent, [src_sent.size(1)])
        pred, attn = self.decoder.decode(hidden_c, output, tgt_vocab.w2i)

        input_ = src_vocab.decode(src_sent.data[0])
        pred = tgt_vocab.decode(pred.data)
        truth = tgt_vocab.decode(tgt_sent.data[0])

        return input_, pred, truth
