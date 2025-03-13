import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultiTaskTranslationDataset(Dataset):
    def __init__(self, en_fr_sentences, fr_sentences, en_es_sentences, es_sentences, tokenizer, max_length=128):
        """
        Dataset for training multilingual translation model.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Creating mixed dataset while keeping correct English pairs
        self.data = []
        
        for en_fr, fr in zip(en_fr_sentences, fr_sentences):
            self.data.append((en_fr, "<fr> " + fr))  # English -> French with language token
        
        for en_es, es in zip(en_es_sentences, es_sentences):
            self.data.append((en_es, "<es> " + es))  # English -> Spanish with language token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, trg_text = self.data[idx]

        src_enc = self.tokenizer(src_text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        trg_enc = self.tokenizer(trg_text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')

        return src_enc['input_ids'].squeeze(0), trg_enc['input_ids'].squeeze(0)


class MultiTaskTrainer:
    def __init__(self, model, train_dataset, valid_dataset, tokenizer, batch_size=32, lr=5e-5, num_epochs=10, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.num_epochs = num_epochs
        self.device = device
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for src_input, trg_input in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
                src_input, trg_input = src_input.to(self.device), trg_input.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(src_input, trg_input[:, :-1])  # Shift for teacher forcing
                loss = self.criterion(output.view(-1, output.size(-1)), trg_input[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
