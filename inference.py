import torch
from model_inference import Transformer
from train import ModelArgs
from pathlib import Path
from tqdm.auto import tqdm
# I will use tiktoken tokenizer instead of sentencepiece
# Since we will have our own training module we can choose
# any tokenizer we want
import tiktoken


class LLaMA:
    def __init__(self, args: ModelArgs):
        self.args = args
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.device = args.device
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size

    # For now code will initiate an untrained model
    def load_model(self, model_path: Path):
        self.model = Transformer(self.args).to(self.device)
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    # Implementation of top p sampling
    def _sample_top_p(self, probs: torch.Tensor, p: float):
        probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_cumsum = probs_sorted.cumsum(dim=-1)
        mask = (probs_cumsum - probs_sorted) > p
        probs_sorted[mask] = 0.0
        probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sorted, num_samples=1)
        next_token = torch.gather(probs_idx, dim=-1, index=next_token)
        return next_token

    # Text generation is done by top p sampling if temperature > 0
    # Otherwise greedy sampling is used
    def generate(
        self,
        prompts: list[str],
        max_generate_len: int = 32,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        prompts_tokens = [self.tokenizer.encode_ordinary(prompt) for prompt in prompts]
        batch_size = len(prompts_tokens)
        assert batch_size <= self.max_batch_size
        max_prompt_length = max([len(tokens) for tokens in prompts_tokens])
        total_len = max_prompt_length + max_generate_len
        assert total_len <= self.max_seq_len

        pad_token = -1
        tokens = torch.full(
            size=(batch_size, total_len),
            fill_value=pad_token,
            dtype=torch.long,
            device=self.device,
        )

        # Populate the tokens with prompt tokens
        for b, t in enumerate(prompts_tokens):
            tokens[b, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        prompt_tokens_mask = tokens != pad_token

        # Generation Loop
        with torch.inference_mode():
            for cur_pos in tqdm(range(total_len - 1), desc='Generating Tokens:'):
                # (batch_size, 1) -> (batch_size, vocab_size)
                logits = self.model(tokens[:, cur_pos : cur_pos + 1], cur_pos).squeeze(
                    dim=1
                )
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    # (batch_size, vocab_size) -> (batch_size, 1)
                    next_tokens = self._sample_top_p(probs, top_p)
                else:
                    # For greedy search we don't need to calculate probs
                    # (batch_size, vocab_size) -> (batch_size, 1)
                    next_tokens = logits.argmax(dim=-1, keepdim=True)

                next_tokens = torch.where(
                    prompt_tokens_mask[:, cur_pos + 1],
                    tokens[:, cur_pos + 1],
                    next_tokens.view(-1),
                )
                tokens[:, cur_pos + 1] = next_tokens

        sample_text = [
            self.tokenizer.decode(t.tolist(), errors="replace") for t in tokens
        ]

        return sample_text


if __name__ == "__main__":
    prompts = [
        "This is a sentence.",
        "This is another sentence but longer.",
        "Let's see what will happen with this one.",
    ]

    args = ModelArgs()
    model = LLaMA(args)
    model.load_model(Path('model/model_1.tar'))

    sampled_text = model.generate(prompts, max_generate_len=100)
    for i, text in enumerate(sampled_text):
        print(f"{text}")
        print("-"*100)

