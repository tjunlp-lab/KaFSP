import json
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import sys
sys.path.append('..')


class DataPrecessForSentence(Dataset):
    def __init__(self, bert_tokenizer, LCQMC_file, max_char_len = 100):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(LCQMC_file)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        
    def get_input(self, file):
        df = pd.read_csv(file)
        tokens_seq = df['description']
        surface_form = df['entity']
        truple_1 = df['truple_1']
        truple_2 = df['truple_2']
        labels = df['label'].values
        tokens_seq = list(map(self.bert_tokenizer.tokenize, tokens_seq))
        surface_form = [str(_) for _ in surface_form]
        surface_form = list(map(self.bert_tokenizer.tokenize, surface_form))
        truple_1 = list(map(self.bert_tokenizer.tokenize, truple_1))
        truple_2 = list(map(self.bert_tokenizer.tokenize, truple_2))
        result = list(map(self.trunate_and_pad, tokens_seq, surface_form, truple_1, truple_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)
    
    def trunate_and_pad(self, tokens_seq, surface_form, truple_1, truple_2):
        if len(tokens_seq) > ((self.max_seq_len - 5)//4):
            tokens_seq = tokens_seq[0:(self.max_seq_len - 5)//4]
        if len(surface_form) > ((self.max_seq_len - 5)//4):
            surface_form = surface_form[0:(self.max_seq_len - 5)//4]
        if len(truple_1) > ((self.max_seq_len - 5)//4):
            truple_1 = truple_1[0:(self.max_seq_len - 5)//4]
        if len(truple_2) > ((self.max_seq_len - 5)//4):
            truple_2 = truple_2[0:(self.max_seq_len - 5)//4]
        seq = seq = tokens_seq + ['<sep>'] + surface_form + ['<sep>'] + truple_1 + ['<sep>'] + truple_2 + ['<sep>'] + ['<cls>']
        seq_segment = [0] * (len(tokens_seq) + 1) + [1] * (len(surface_form) + 1) + [2] * (len(truple_1)+1) + [3]* (len(truple_2)+1) + [4]
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (self.max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = seq_segment + padding
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment