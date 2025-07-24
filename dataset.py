import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from config import cfg


class Dataset(nn.Module):
    def __init__(self, tokenizer, txt, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def dataloader(txt, batch_size, max_length, stride, shuffle, drop_last, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = Dataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("data.txt", "r", encoding="") as f:
    data = f.read()

train_ratio = 0.9
split = int(train_ratio * len(data))
train_data = data[:split]
val_data = data[split:]

train_dataloader = dataloader(
    train_data,
    batch_size=2,
    max_length=cfg["context_length"],
    stride=cfg["stride"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = dataloader(
    val_data,
    batch_size=2,
    max_length=cfg["context_length"],
    stride=cfg["stride"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)