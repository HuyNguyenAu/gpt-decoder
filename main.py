import math
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
from time import time
import json

# Hyperparams.
batch_size = 16  # The number of independent sequences to process in parallel.
block_size = 32  # The length of a sequence or the context size.
steps = 5000  # The number of steps to train for.
eval_every_steps = 200  # The epoch interval to print losses.
learning_rate = 1e-3  # The learning rate.
n_layers = 6  # The number of attention heads.
n_embed = 512  # The number of embedding dimensions.
n_heads = 8  # The number of communication channels
dropout_rate = 0.0  # The drop out rate.
device = 'cpu'  # The device to run the training on.

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'

# For reproducibility.
torch.manual_seed(1337)

# Load the text.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Here we're using a simple tokeniser which results in a longer list of tokens
# for a given string. In practise, more sophisticated ones are used such as
# WordPiece which outputs a much shorter list of tokens.

# Convert a string to a list of tokens.
stoi = {char: i for i, char in enumerate(chars)}
def encode(string): return [stoi[char] for char in string]


# Convert a list of token to a string.
itos = {i: char for i, char in enumerate(chars)}
def decode(tokens): return ''.join([itos[token] for token in tokens])


# Create training and validation datasets.
data = torch.tensor(encode(text), dtype=torch.long)
dataset_length = int(0.9 * len(data))
train_data = data[:dataset_length]
val_data = data[dataset_length:]


def get_batch(split):
    '''
    Get the inputs and targets as a rolling window up to the context size.
    Inputs:
        torch.Size([4, 8])
        tensor([[24, 43, 58,  5, 57,  1, 46, 43],
                [44, 53, 56,  1, 58, 46, 39, 58],
                [52, 58,  1, 58, 46, 39, 58,  1],
                [25, 17, 27, 10,  0, 21,  1, 54]])
        Targets:
        torch.Size([4, 8])
        tensor([[43, 58,  5, 57,  1, 46, 43, 39],
                [53, 56,  1, 58, 46, 39, 58,  1],
                [58,  1, 58, 46, 39, 58,  1, 46],
                [17, 27, 10,  0, 21,  1, 54, 39]])
        ----
        When input is [24] the target: 43
        When input is [24, 43] the target: 58
        When input is [24, 43, 58] the target: 5
        When input is [24, 43, 58, 5] the target: 57
        When input is [24, 43, 58, 5, 57] the target: 1
        When input is [24, 43, 58, 5, 57, 1] the target: 46
        When input is [24, 43, 58, 5, 57, 1, 46] the target: 43
        When input is [24, 43, 58, 5, 57, 1, 46, 43] the target: 39
        When input is [44] the target: 53
        When input is [44, 53] the target: 56
        When input is [44, 53, 56] the target: 1
        When input is [44, 53, 56, 1] the target: 58
        ...
        When input is [25, 17, 27, 10, 0] the target: 21
        When input is [25, 17, 27, 10, 0, 21] the target: 1
        When input is [25, 17, 27, 10, 0, 21, 1] the target: 54
        When input is [25, 17, 27, 10, 0, 21, 1, 54] the target: 39
    '''

    data = train_data if split == 'train' else val_data
    # Generate a list of random numbers as the starting indexes
    # into the text.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # From the list of starting indexes, build a list of
    # sequences which contain the characters from the starting index
    # and to starting index + context size.
    x = torch.stack([data[i:i + block_size] for i in ix])
    # The targets are just the following character.
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    # Copy to device.
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    '''
    Estimate the training and validation loss..
    '''

    out = {}

    # Set model in evaluation mode.
    # Disables layers like dropout and layer norm.
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_every_steps)

        # Get the average loss for every eval_every_steps steps.
        for k in range(eval_every_steps):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    # Set model to training mode.
    model.train()

    return out


class MaskedMultiHeadAttention(nn.Module):
    '''
    Multiple self attention heads in parallel.
    '''

    def __init__(self, n_heads: int, n_embed: int, dropout_rate: float):
        super().__init__()

        assert n_embed % n_heads == 0, f'Embedding dimension {n_embed} should be divisible by the number of heads {n_heads}.'

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate

        # Key: What features I have that may be interesting to other tokens.
        # Query: What features am I interested/looking for in other tokens.
        # Value: What features am I interested/looking for in other tokens and that may be interesting to other tokens. It will also tell others that here's what I will communicate if you find me interesting.

        # Key, query, value batched projections for all heads.
        # Stack the K, Q, and V projections.
        self.attention = nn.Linear(
            in_features=n_embed, out_features=n_embed * 3, bias=False)
        self.proj = nn.Linear(in_features=n_embed, out_features=n_embed)
        self.attention_dropout = nn.Dropout(p=dropout_rate)
        self.residual_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape  # (B, T, C) = (batch_size, block_size, n_embed).

        # Since the output of attention is (B, T, C * 3), we split it into 3 equal parts of (B, T, C).
        q, k, v = self.attention(x).split(self.n_embed, dim=2)

        # Since the scaled dot product attention expects the input to be (B, ..., T, C), we need to reshape the output of the attention.
        # This attention has the K, Q, V with the heads embedded in the channel dimension.
        # The data is of (B, T, C) shape and we want to squeeze the channel dimension into heads and head channels.
        # That is C === self.n_heads * (C // self.n_heads).
        # (B, n_heads, T, block_size)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # (B, n_heads, T, block_size)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # (B, n_heads, T, block_size)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # The Pytorch implementation of the scaled dot product attention returns (B, n_heads, block_size, block_size).
        # Q: (B, n_heads, T, block_size) x K: (B, n_heads, block_size, T) -> (B, n_heads, T, T).
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True)
        # Stretch the output back into (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Apply projection.
        y = self.residual_dropout(self.proj(y))

        return y


class GeLU(nn.Module):
    '''
    The GeLU layer.
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FeedForward(nn.Module):
    '''
    A simple FFN.
    '''

    def __init__(self, n_embed: int, dropout_rate: float) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # Apply Position-wise FFN multiplier.
            nn.Linear(in_features=n_embed, out_features=4 * n_embed),
            GeLU(),
            # Apply Position-wise FFN multiplier.
            nn.Linear(in_features=4 * n_embed, out_features=n_embed),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    '''
    The attention block.
    '''

    def __init__(self, n_heads: int, n_embed: int, dropout_rate: float) -> None:
        super().__init__()

        # The self attention heads. Means the K, V, and Q are from the same source.
        self.sa = MaskedMultiHeadAttention(
            n_heads=n_heads, n_embed=n_embed, dropout_rate=dropout_rate)
        # A simple indirection layer.
        self.ffwd = FeedForward(
            n_embed=n_embed, dropout_rate=dropout_rate)
        # self.ffwd = FastKAN([n_embed * 28, 64, n_embed])
        self.ln1 = nn.LayerNorm(normalized_shape=n_embed)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embed)

    def forward(self, x: torch.Tensor):
        # Apply pre-norm formulation where the layer norm is applied before each layer.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class PositionEmbedding(nn.Module):
    '''
    The position embedding.
    '''

    def __init__(self, block_size: int, n_embed: int):
        super().__init__()

        self.block_size = block_size
        self.n_embed = n_embed
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=n_embed, bias=False),
            GeLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((1, self.n_embed), device=device)
        for i in range(self.block_size - 1):
            et = x[i].view(1)
            etn = torch.unsqueeze(self.net(et), dim=0)
            out = torch.cat((out, etn), dim=0)

        return out


class GPTDecoder(nn.Module):
    '''
    The GPT decoder.
    '''

    def __init__(self, n_heads: int, n_embed: int, n_layers: int, dropout_rate: float, vocab_size: int, block_size: int):
        super().__init__()

        # Each token directly reads off the logits for the next token
        # from the lookup table.
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embed)
        # Each position from 0 to block_size - 1 will have it's own embedding table.
        # self.position_embedding_table = nn.Embedding(
        #     num_embeddings=block_size, embedding_dim=n_embed)
        self.position_embedding_table = PositionEmbedding(block_size=block_size, n_embed=n_embed)

        # self.position_embedding =

        # 4 communication channels of 8-dimension self attention in parallel.
        self.blocks = nn.Sequential(
            *[Block(n_heads=n_heads, n_embed=n_embed, dropout_rate=dropout_rate) for _ in range(n_layers)])
        # Final layer norm.
        self.lnf = nn.LayerNorm(normalized_shape=n_embed)
        # Add a layer of indirection on top of the embedding table.
        self.lm_head = nn.Linear(in_features=n_embed, out_features=vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        device = idx.device
        B, T = idx.shape

        assert T <= self.block_size, f'The number of tokens {T} should be less than or equal to the block size {self.block_size}.'

        # idx and targets are both (B, T) tensors of ints.
        # (B, T, C) = (Batch, Time, Channel) = (batch_size, block_size, vocab_size)
        token_emb: torch.Tensor = self.token_embedding_table(
            idx)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(
            torch.arange(T*1.0, device=device))  # (T, n_embed)

        # print("idx", idx.shape)
        # print("token_emb", token_emb.shape)
        # print("pos_emb", pos_emb.shape)
        # print("arange", torch.arange(T).shape)

        x = token_emb + pos_emb  # (B, T, n_embed)
        # Apply the heads of self attention (B, T, n_embed).
        x = self.blocks(x)
        x = self.lnf(x)  # (B, T, n_embed)
        logits: torch.Tensor = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # F.cross_entropy expects a tensor to be (B, C, T) but as a 2D tensor (B * T, C), where B * T is the minibatch.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is array of (B, T) of indicies in the current context.
        for _ in range(max_new_tokens):
            # Make sure idx is cropped to the last block_size tokens.
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions.
            logits, _ = self(idx_cond)
            # Focus only on the last time step.
            logits = logits[:, -1, :]  # (B, C)
            # Get probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


# Move the model to the GPU.
model = GPTDecoder(n_heads=n_heads, n_embed=n_embed, n_layers=n_layers,
                   dropout_rate=dropout_rate, vocab_size=vocab_size, block_size=block_size)
m = model.to(device)

# Print the number of parameters in the model.
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Set the optimiser.
optimiser = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, fused=device == "cuda")

# Get the current datetime.
current_date = datetime.now()

# Keep track of the time elasped.
start_time = time()

# Train the model.
for step in range(steps):
    # Print out losses and store them in a file.
    if step % eval_every_steps == 0 or step == steps - 1:
        losses = estimate_loss()
        eval = {
            'step': step,
            'train_loss': round(losses['train'].item(), 4),
            'val_loss': round(losses['val'].item(), 4),
            'time_elapsed': round(time() - start_time, 4)
        }
        print(eval)

        with open(f'{current_date.isoformat()}.json', 'a', encoding='utf-8') as file:
            json.dump(eval, file)
            file.write('\n')

    # Sample a batch of data.
    xb, yb = get_batch('train')

    # Evaluate the loss.
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# Sample from the model.
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
