import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, df):
        self.df = df
        if {"center", "context"}.issubset(df.columns):
            self.center_col = "center"
            self.context_col = "context"
        elif {"center_idx", "context_idx"}.issubset(df.columns):
            self.center_col = "center_idx"
            self.context_col = "context_idx"
        elif {"input", "target"}.issubset(df.columns):
            self.center_col = "input"
            self.context_col = "target"
        else:
            raise KeyError(f"Unrecognized skipgram_df columns: {list(df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center = int(row[self.center_col])
        context = int(row[self.context_col])
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, centers, contexts, negatives):
        v = self.in_embed(centers)
        u_pos = self.out_embed(contexts)
        u_neg = self.out_embed(negatives)

        pos_score = torch.sum(v * u_pos, dim=1)
        pos_logprob = torch.log(torch.sigmoid(pos_score) + 1e-9)

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_logprob = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-9), dim=1)

        loss = -(pos_logprob + neg_logprob).mean()
        return loss

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
vocab_size = len(data["idx2word"])


# Precompute negative sampling distribution below
counter = data.get("counter", None)
idx2word = data["idx2word"]

counts = torch.ones(vocab_size, dtype=torch.float)
if counter is not None:
    for i, w in enumerate(idx2word):
        counts[i] = float(counter.get(w, 1))

neg_probs = torch.pow(counts, 0.75)
neg_probs = neg_probs / neg_probs.sum()


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_targets(center, context, vocab_size):
    return context

# Training loop
model.train()
for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

    for centers, contexts in pbar:
        centers = centers.to(device)
        contexts = contexts.to(device)

        negatives = torch.multinomial(
            neg_probs.to(device),
            num_samples=centers.size(0) * NEGATIVE_SAMPLES,
            replacement=True,
        ).view(centers.size(0), NEGATIVE_SAMPLES)

        optimizer.zero_grad(set_to_none=True)
        loss = model(centers, contexts, negatives)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(1, len(dataloader))
    print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")


# Save embeddings and mappings
# embeddings = model.get_embeddings()
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
