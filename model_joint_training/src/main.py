from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn

import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm

class Manager():
    def __init__(self, is_train=True, ckpt_name=None):
        # Load vocabs
        print("Loading vocabs...")
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{joint_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word
            self.trg_i2w[i] = word  # Use the same vocabulary for both

        print(f"The size of vocab is {len(self.src_i2w)}.")

        # Load Transformer model & Adam optimizer
        print("Loading Transformer model & Adam optimizer...")
        #self.model = Transformer(src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)).to(device)
        self.model = Transformer(vocab_size=len(self.src_i2w)).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = sys.float_info.max

        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(TRAIN_NAME)
            self.valid_loader = get_data_loader(VALID_NAME)

        print("Setting finished.")

    def train(self):
        print("Training starts.")

        for epoch in range(1, num_epochs+1):
            self.model.train()
            train_losses = []
            start_time = datetime.datetime.now()

            for i, batch in tqdm(enumerate(self.train_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)
                output = self.model(src_input, trg_input, e_mask, d_mask)

                trg_output_shape = trg_output.shape
                self.optim.zero_grad()
                loss = self.criterion(
                    output.view(-1, len(self.trg_i2w)),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                loss.backward()
                self.optim.step()
                train_losses.append(loss.item())
                
                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

            end_time = datetime.datetime.now()
            mean_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch}: Train loss {mean_train_loss:.4f}")
            
            valid_loss, valid_time = self.validation()
            
            if valid_loss < self.best_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss
                }
                torch.save(state_dict, f"{ckpt_dir}/best_ckpt.tar")
                print("***** Current best checkpoint is saved. *****")

        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
        valid_losses = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)
                e_mask, d_mask = self.make_mask(src_input, trg_input)
                output = self.model(src_input, trg_input, e_mask, d_mask)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, len(self.trg_i2w)),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )
                valid_losses.append(loss.item())
        mean_valid_loss = np.mean(valid_losses)
        valid_time = "N/A"
        return mean_valid_loss, valid_time

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)
        d_mask = (trg_input != pad_id).unsqueeze(1)
        nopeak_mask = torch.tril(torch.ones([1, seq_len, seq_len], dtype=torch.bool)).to(device)
        d_mask = d_mask & nopeak_mask
        return e_mask, d_mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
    args = parser.parse_args()

    if args.mode == 'train':
        manager = Manager(is_train=True, ckpt_name=args.ckpt_name)
        manager.train()
    elif args.mode == 'inference':
        print("Inference mode not yet implemented for multi-task training.")