from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from constants import *

import torch
import sentencepiece as spm
import numpy as np

# Load joint SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(f"{SP_DIR}/{joint_model_prefix}.model")

# import os
# abs_sp_model_path = os.path.abspath(f"{SP_DIR}/{joint_model_prefix}.model")
# sp.Load(abs_sp_model_path)


def get_data_loader(file_name):
    print(f"Getting source/target {file_name}...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r') as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r') as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_text(src_text_list, is_target=False) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_text(trg_text_list, is_target=True) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_text(text_list, is_target=False):
    tokenized_list = []
    input_tokenized_list = []
    output_tokenized_list = []

    for text in tqdm(text_list):
        tokenized = sp.EncodeAsIds(text.strip())
        if is_target:
            trg_input = [sos_id] + tokenized
            trg_output = tokenized + [eos_id]
            input_tokenized_list.append(pad_or_truncate(trg_input))
            output_tokenized_list.append(pad_or_truncate(trg_output))
        else:
            tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    if is_target:
        return input_tokenized_list, output_tokenized_list
    return tokenized_list

class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]