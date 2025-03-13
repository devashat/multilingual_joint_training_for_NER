'''
HOW TO RUN:

python3 finetuning.py --ckpt_name ../saved_model/best_ckpt.tar --epochs 20
'''


import torch
import os
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from torch import nn
from constants import *
from custom_data import *
from transformer import Transformer

class FineTuner:
    def __init__(self, ckpt_name):
        print("Loading vocabulary...")
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{joint_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word
            self.trg_i2w[i] = word  # Using the same vocabulary

        print(f"Vocab size: {len(self.src_i2w)}")
        print("Loading Transformer model...")
        self.model = Transformer(vocab_size=len(self.src_i2w)).to(device)
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.best_loss = float('inf')

        if ckpt_name:
            ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../joint_training_saved_model", ckpt_name))
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, weights_only=False) ## had to set to weights only false
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_loss = checkpoint['loss']
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")


        print("Loading train fine-tuning data from data file...")
        # Load the train data
        with open(f"{DATA_DIR}/fine_tune_data/train.txt", 'r', encoding="utf-8") as f:
            train_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Separate sources (English with BIO) and targets (<fr>/<es> translations) for train
        src_train_list = []
        trg_train_list = []
        for i in range(0, len(train_lines), 2):  # Every two lines (source -> target)
            src_train_list.append(train_lines[i])   # English with BIO
            trg_train_list.append(train_lines[i + 1])  # French/Spanish translation
    
        # Tokenize & process train
        print("Tokenizing & Padding source train data...")
        train_src_tokenized_list = process_text(src_train_list, is_target=False)
        print("Tokenizing & Padding target train data...")
        train_input_trg_list, train_output_trg_list = process_text(trg_train_list, is_target=True)

        # Create dataset & dataloaders
        train_dataset = CustomDataset(train_src_tokenized_list, train_input_trg_list, train_output_trg_list)

        print("Loading val fine-tuning data from data file...")
        # Load the val data
        with open(f"{DATA_DIR}/fine_tune_data/val.txt", 'r', encoding="utf-8") as f:
            val_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Separate sources (English with BIO) and targets (<fr>/<es> translations) for val
        src_val_list = []
        trg_val_list = []
        for i in range(0, len(val_lines), 2):  # Every two lines (source -> target)
            src_val_list.append(val_lines[i])   # English with BIO
            trg_val_list.append(val_lines[i + 1])  # French/Spanish translation

        # Tokenize & process val
        print("Tokenizing & Padding source val data...")
        val_src_tokenized_list = process_text(src_val_list, is_target=False)
        print("Tokenizing & Padding target val data...")
        val_input_trg_list, val_output_trg_list = process_text(trg_val_list, is_target=True)

        # Create dataset & dataloaders
        val_dataset = CustomDataset(val_src_tokenized_list, val_input_trg_list, val_output_trg_list)


        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Use same dataset for validation

        print(f"Loaded {len(train_dataset)} train samples for fine-tuning.")
        print(f"Loaded {len(val_dataset)} val samples for fine-tuning.")



    def train(self, num_epochs=5):
        print("Starting fine-tuning...")
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_losses = []
            start_time = datetime.datetime.now()

            for _, batch in tqdm(enumerate(self.train_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)
                output = self.model(src_input, trg_input, e_mask, d_mask)
                loss = self.criterion(
                    output.view(-1, len(self.trg_i2w)),
                    trg_output.view(-1)
                )
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_losses.append(loss.item())
                torch.cuda.empty_cache()

            mean_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch}: Train loss = {mean_train_loss:.4f}")
            valid_loss = self.validate()

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.save_checkpoint()
                print("New best model saved.")
        print("Fine-tuning complete!")

    def validate(self):
        print("Running validation...")
        self.model.eval()
        valid_losses = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)
                e_mask, d_mask = self.make_mask(src_input, trg_input)
                output = self.model(src_input, trg_input, e_mask, d_mask)
                loss = self.criterion(
                    output.view(-1, len(self.trg_i2w)),
                    trg_output.view(-1)
                )
                valid_losses.append(loss.item())
        mean_valid_loss = np.mean(valid_losses)
        print(f"Validation Loss: {mean_valid_loss:.4f}")
        return mean_valid_loss

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)
        d_mask = (trg_input != pad_id).unsqueeze(1)
        nopeak_mask = torch.tril(torch.ones([1, seq_len, seq_len], dtype=torch.bool)).to(device)
        d_mask = d_mask & nopeak_mask
        return e_mask, d_mask

    def save_checkpoint(self):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'loss': self.best_loss
        }, f"{ckpt_dir}/finetuned_best_ckpt.tar")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_name', required=True, help="Checkpoint file to fine-tune from")
    parser.add_argument('--epochs', type=int, default=5, help="Number of fine-tuning epochs")
    args = parser.parse_args()

    finetuner = FineTuner(ckpt_name=args.ckpt_name)
    finetuner.train(num_epochs=args.epochs)
