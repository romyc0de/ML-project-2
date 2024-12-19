import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Paths to files
train_embeddings_file = "embeddings/bert_embeddings.npy"
test_embeddings_file = "embeddings/bert_test_embeddings.npy"
pos_file = "twitter-datasets/train_pos.txt"
neg_file = "twitter-datasets/train_neg.txt"

# Read tweets from a file
def read_tweets(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Read positive and negative training data
print("Reading training data...")
pos_tweets = read_tweets(pos_file)
neg_tweets = read_tweets(neg_file)

# Generate labels
labels = np.concatenate([
    np.ones(len(pos_tweets), dtype=int),
    np.zeros(len(neg_tweets), dtype=int)
])

# Load precomputed embeddings
print("Loading precomputed training embeddings...")
cls_embeddings_np = np.load(train_embeddings_file)

# Validate consistency
assert len(labels) == cls_embeddings_np.shape[0], "Mismatch in number of labels and embeddings."

# Normalize embeddings
embedding_scaler = StandardScaler()
cls_embeddings_np = embedding_scaler.fit_transform(cls_embeddings_np)

# Split data
print("Splitting data into train and validation...")
X_train, X_val, y_train, y_val = train_test_split(
    cls_embeddings_np, labels, test_size=0.2, stratify=labels, random_state=42
)

# Custom Dataset class for PyTorch
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Create DataLoaders
batch_size = 64
train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset = EmbeddingDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rates):
        super(SentimentClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate in zip(layer_sizes, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define model
input_dim = cls_embeddings_np.shape[1]
layer_sizes = [512, 256, 128]
dropout_rates = [0.3, 0.3, 0.3]

model = SentimentClassifier(
    input_dim=input_dim,
    layer_sizes=layer_sizes,
    dropout_rates=dropout_rates
)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3

from torch.nn import CrossEntropyLoss
class_weights = torch.tensor([len(neg_tweets) / len(pos_tweets), 1.0], dtype=torch.float32).to(device)
criterion = CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
from transformers import get_scheduler
scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * epochs)

# Training loop

patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

train_losses, val_losses = [], []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    train_loss = 0
    for embeddings, labels in tqdm(train_loader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for embeddings, labels in tqdm(val_loader, desc="Validation"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Validation loss improved. Saved best model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epochs.")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")
plt.show()

# Load best model
model.load_state_dict(torch.load("best_model.pth"))


# Predict on test set
print("Loading test embeddings...")
test_embeddings = np.load(test_embeddings_file)
test_embeddings = embedding_scaler.transform(test_embeddings)  # Normalize test embeddings

test_dataset = EmbeddingDataset(test_embeddings, np.zeros(len(test_embeddings)))  # Dummy labels
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Predicting on test set...")
test_preds = []
test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
with torch.no_grad():
    for embeddings, _ in test_loader_tqdm:
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())

# Save predictions
print("Saving predictions...")
submission_df = pd.DataFrame({
    "Id": np.arange(1, len(test_preds) + 1),
    "Prediction": np.where(np.array(test_preds) == 1, 1, -1)  # Convert to required format
})

submission_df.to_csv("Submissions/submission_BERT_nn-small.csv", index=False)
print("Predictions saved to 'Submissions/submission_BERT_nn.csv'.")
