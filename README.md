# Sequence to Sequence with Attetion for Machine Translation

## Model
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## How to run
```
glt clone https://github.com/sobamchan/translation_seq2seq_pytorch.git
cd translation_seq2seq_pytorch
glt clone https://github.com/odashi/small_parallel_enja.git
mkdir test
python main.py --gpu-id <YOUR GPU ID>
```

Check main.py for more options.
