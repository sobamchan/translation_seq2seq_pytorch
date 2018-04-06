from collections import Counter


class Vocabulary:

    def __init__(self, sents, args):
        self.sents = sents
        if hasattr(args, 'vocab_size'):
            self.vocab_size = args.vocab_size
        else:
            self.vocab_size = False

        vocab, w2i, i2w = self.build_vocab()
        self.vocab = vocab
        self.w2i = w2i
        self.i2w = i2w

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self):
        vocab_counter = Counter()
        for sent in self.sents:
            for word in sent.split():
                vocab_counter[word] += 1

        if self.vocab_size:
            vocab = [w for w, _ in vocab_counter.most_common(self.vocab_size)]
        else:
            vocab = list(vocab_counter.keys())

        w2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
        vocab += list(w2i.keys())
        for word in vocab:
            if word not in w2i.keys():
                w2i[word] = len(w2i)
        i2w = {idx: w for w, idx in w2i.items()}
        assert len(vocab) == len(w2i) == len(i2w)
        return vocab, w2i, i2w

    def encode(self, sent):
        w2i = self.w2i
        vocab = self.vocab
        words = sent.split()
        encoded_words = []
        for w in words:
            wid = w2i[w] if w in vocab else w2i['<UNK>']
            encoded_words.append(wid)
        return encoded_words, words
