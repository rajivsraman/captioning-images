import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import EncoderCNN, DecoderWithAttention
from dataset import FlickrDataset, Vocabulary, MyCollate, get_transforms

# Hyperparameters
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
FREQ_THRESHOLD = 5
ATTENTION_DIM = 256

# Paths
MODEL_PATH = "best_caption_model.pth"
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, vocab):
    encoder.train()
    decoder.train()
    epoch_loss = 0

    for imgs, captions in tqdm(loader):
        imgs, captions = imgs.to(device), captions.to(device)
        features = encoder(imgs)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate_model(encoder, decoder, loader, criterion, device, vocab):
    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, captions in loader:
            imgs, captions = imgs.to(device), captions.to(device)
            features = encoder(imgs)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            val_loss += loss.item()

    return val_loss / len(loader)


def train_model():
    df = pd.read_csv(CAPTION_FILE)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build_vocabulary(train_df["caption"].tolist())

    transform = get_transforms()

    train_data = FlickrDataset(train_df, vocab, IMAGE_DIR, transform)
    val_data = FlickrDataset(val_df, vocab, IMAGE_DIR, transform)

    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MyCollate(pad_idx))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MyCollate(pad_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(EMBED_SIZE, HIDDEN_SIZE, len(vocab), ATTENTION_DIM).to(device)

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device, vocab)
        val_loss = validate_model(encoder, decoder, val_loader, criterion, device, vocab)

        train_losses.append(train_loss) 
        val_losses.append(val_loss)    

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": vocab.stoi
            }, MODEL_PATH)
            print("Saved best model.")

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    print("Saved loss plot to loss_plot.png")


if __name__ == "__main__":
    train_model()
