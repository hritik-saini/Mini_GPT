import torch
from model.decoder import Decoder

# Hyperparameters
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 65
n_head = 6
n_layer = 6
dropout = 0.2

# -------------

torch.manual_seed(1337)

# Read the data File
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create the unique Vocabulary of characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from character to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the data into tensor array
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data in test and train
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Define the Batch data processing
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


# Define the estimate loss - Mean over all the eval iterations
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initiate the Model object
model = Decoder(vocab_size, n_embd, block_size, dropout, n_layer, n_head)

# # Define the Adam optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for iter in range(max_iters):
#
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# # Save the model weights
# torch.save(model.state_dict(), 'model_weight_with_single_head_final.pth')
