import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from model_train import Transformer
import requests
import tiktoken
from tqdm.auto import tqdm
from argparse import ArgumentParser


@dataclass
class ModelArgs:
    dim: int = 256
    vocab_size: int = 50304
    batch_size: int = 8
    max_batch_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = 2
    seq_len: int = 256
    max_seq_len: int = 256
    rms_eps: float = 1e-6
    head_dim: int = dim // n_heads
    device: str = "cuda"


class CustomDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        # total_length +1 of total number of tokens because
        # buffer is seq_len + 1
        total_length = len(self.tokens) + 1
        remainder = total_length % self.seq_len
        dividend = total_length // self.seq_len
        return dividend + 1 if remainder else dividend

    def __getitem__(self, index):
        buff = self.tokens[index * self.seq_len : (index + 1) * self.seq_len + 1]
        if len(buff) < self.seq_len + 1:
            buff = self.tokens[-self.seq_len - 1 :]
        x, y = (
            torch.tensor(buff[:-1], dtype=torch.long),
            torch.tensor(buff[1:], dtype=torch.long),
        )
        return x, y


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=12)
    train_args = parser.parse_args()

    args = ModelArgs()

    content = requests.get(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    text = content.text

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode_ordinary(text)

    dataset = CustomDataset(tokens, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Transformer(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    model.train()
    for epoch in tqdm(range(1, train_args.num_epochs+1)):
        loss_epoch = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            x, y = x.to(args.device), y.to(args.device)
            batch_size, seq_len = x.size()
            logits = model(x)
            # (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(batch_size * seq_len, -1), y.view(-1))
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss_epoch / len(dataloader):.4f}")

    torch.save({'model_state_dict': model.state_dict()}, "model/model_1.tar")