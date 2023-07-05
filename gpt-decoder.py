import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams.
batch_size = 16  # The number of independent sequences to process in parallel.
block_size = 32  # The length of a sequence or the context size.
steps = 5000  # The number of steps to train for.
eval_every_steps = 200  # The epoch interval to print losses.
learning_rate = 1e-3  # The learning rate.
n_embed = 64  # The number of embedding dimensions.
n_head = 4 # The number of communication channels
n_layer = 4 # The number of attention heads.
dropout_rate = 0.0 # The drop out rate.
device = 'cpu' # The device to run the training on.

# if torch.cuda.is_available():
#     device = 'cuda'
# elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = 'mps'

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
    data = train_data if split == 'train' else val_data
    # Generate a list of random numbers as the starting indexes
    # into the text.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # From the list of starting indexes, build a list of
    # sequences which contain the characters from the starting index
    # and to starting index + context size.
    x = torch.stack([data[i:i + block_size] for i in ix])
    # The targets are just the following character .
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    # Copy to device.
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
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


class Head(nn.Module):
    '''
    A single causal self attention head.
    '''

    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout_rate: int):
        super().__init__()

        # What features I have that may be interesting to other tokens.
        self.key = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        # What features am I interested/looking for in other tokens.
        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        # What features am I interested/looking for in other tokens and that may be interesting to other tokens. It will also tell others that here's what I will communicate if you find me interesting.
        self.value = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Apply Xavier uniform initialisation according to T-Fixup.
        nn.init.xavier_uniform_(self.key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.query.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.value.weight, gain=1 / math.sqrt(2))

    def forward(self, x: torch.Tensor):
        _, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd).
        # Create the key and query in the (B, T) arrangement for each token.
        k: torch.Tensor = self.key(x)  # (B, T, head_size)
        q: torch.Tensor = self.query(x)  # (B, T, head_size)
        # Here each batch will have different weights because there are different tokens.
        # Each token in wei, knows it's position and what it has (key) and is looking for (query).
        # Each channel in C knows what it is (I am a constantent, I am a letter, .etc) which means that specific key
        # in that channel will have a key that will have a higher number. When a query is dot prod with that key it will
        # create a high affinity.
        # (B, T, head_size) @ (B, head_size, T) => (B, T, T)
        wei: torch.Tensor = q @ k.transpose(-2, -1) * C**-0.5
        # Here we are scaling the attention so that the q, k, and wei will be unit variance.
        # Here we are not allowing all tokens to talk to each other. An encoder block will not have this line.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)  # type: ignore
        # Softmaxing will create higher probs due to higher affinities above.
        wei = F.softmax(input=wei, dim=-1) # (B, T, T)
        # Add dropout layer.
        wei = self.dropout(wei)
        # Here x can be throught of as private info to this current token.
        v: torch.Tensor = self.value(x) # (B, T, C).
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MaskedMultiHeadAttention(nn.Module):
    '''
    Multiple self attention heads in parallel.
    '''

    def __init__(self, n_heads: int, head_size: int, n_embed: int, block_size: int, dropout_rate: int):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size=head_size, n_embed=n_embed, block_size=block_size, dropout_rate=dropout_rate) for _ in range(n_heads)])
        self.proj = nn.Linear(in_features=n_embed, out_features=n_embed)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Apply Xavier uniform initialisation according to T-Fixup.
        nn.init.xavier_uniform_(self.proj.weight, gain=1 / math.sqrt(2))

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout_rate: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_embed, out_features=4 * n_embed),
            nn.ReLU(),
            nn.Linear(in_features=4 * n_embed, out_features=n_embed),
            nn.Dropout(p=dropout_rate),
        )
        
        # Apply Xavier uniform initialisation according to T-Fixup.
        nn.init.xavier_uniform_(self.net[0].weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.net[-2].weight, gain=1 / math.sqrt(2))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, block_size: int, dropout_rate: int) -> None:
        super().__init__()

        head_size = n_embed // n_heads
        # The self attention heads. Means the K, V, and Q are from the same source.
        self.sa = MaskedMultiHeadAttention(n_heads=n_heads, head_size=head_size, n_embed=n_embed, block_size=block_size, dropout_rate=dropout_rate)
        # A simple indirection layer.
        self.ffwd = FeedForward(n_embed=n_embed, dropout_rate=dropout_rate)
        self.ln1 = nn.LayerNorm(normalized_shape=n_embed)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embed)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTDecoder(nn.Module):
    def __init__(self, n_embed: int, block_size: int, n_heads: int, n_layer: int, dropout_rate: int):
        super().__init__()

        # Each token directly reads off the logits for the next token
        # from the lookup table.
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        # Each position from 0 to block_size - 1 will have it's own embedding table.
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)
        # 4 communication channels of 8-dimension self attention in parallel.
        self.blocks = nn.Sequential(
            *[Block(n_heads=n_heads, n_embed=n_embed, block_size=block_size, dropout_rate=dropout_rate) for _ in range(n_layer)])
        # Final layer norm.
        self.lnf = nn.LayerNorm(normalized_shape=n_embed)
        # Add a layer of indirection on top of the embedding table.
        self.lm_head = nn.Linear(in_features=n_embed, out_features=vocab_size)
        self.block_size = block_size
        
        # Apply Xavier uniform initialisation according to T-Fixup.
        nn.init.xavier_uniform_(self.lm_head.weight, gain=1 / math.sqrt(2))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of ints.
        # (B, T, C) = (Batch, Time, Channel) = (batch_size, block_size, vocab_size)
        token_emb: torch.Tensor = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # Apply the heads of self attention (B, T, C).
        x = self.lnf(x)  # (B,T,C)
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
model = GPTDecoder(n_embed=n_embed, block_size=block_size, n_heads=n_head, n_layer=n_layer, dropout_rate=dropout_rate)
m = model.to(device)

# Print the number of parameters in the model.
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Set the optimiser.
# optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimiser = torch.optim.NAdam(model.parameters(), lr=learning_rate)

# Train the model.
for step in range(steps):
    # Print out losses.
    if step % eval_every_steps == 0 or step == steps - 1:
        losses = estimate_loss()
        print(
            f"Step: {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

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
