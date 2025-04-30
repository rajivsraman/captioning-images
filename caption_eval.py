import os
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from model import EncoderCNN, DecoderWithAttention
from dataset import Vocabulary, get_transforms
from PIL import Image
import nltk
nltk.download('punkt')

# Paths and settings
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
MODEL_PATH = "best_caption_model.pth"
MAX_LEN = 20

# Load model components
def load_model(vocab_size, device):
    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(
        embed_size=256,
        hidden_size=512,
        vocab_size=vocab_size,
        attention_dim=256
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    return encoder, decoder, checkpoint['vocab']

# Generate caption for an image
def generate_caption(image_path, encoder, decoder, vocab, device):
    transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        features = encoder(image)
        outputs = []
        word = torch.tensor([vocab['<START>']]).to(device)
        h, c = (torch.zeros(1, 512).to(device), torch.zeros(1, 512).to(device))

        for _ in range(MAX_LEN):
            embedded = decoder.embedding(word).squeeze(1)
            context, _ = decoder.attention(features, h)
            lstm_input = torch.cat([embedded, context], dim=1)
            h, c = decoder.lstm(lstm_input, (h, c))
            output = decoder.fc(h)
            word = output.argmax(1).unsqueeze(1)

            predicted_word = vocab_inv[word.item()]
            if predicted_word == "<END>":
                break
            outputs.append(predicted_word)

    return ' '.join(outputs)

# Evaluate BLEU score on test data

def evaluate_bleu(full_df, encoder, decoder, vocab, device):
    global vocab_inv
    vocab_inv = {idx: token for token, idx in vocab.items()}

    # Collect all references (all captions per image)
    references = full_df.groupby('image')['caption'].apply(lambda caps: [nltk.word_tokenize(c.lower()) for c in caps]).to_dict()

    # Sample 10% of image IDs for evaluation
    image_ids = list(references.keys())
    sampled_ids = pd.Series(image_ids).sample(frac=0.1, random_state=42).tolist()

    smoothie = SmoothingFunction().method4
    scores = []

    for image_id in sampled_ids:
        path = os.path.join(IMAGE_DIR, image_id)
        generated = generate_caption(path, encoder, decoder, vocab, device)
        hypothesis = nltk.word_tokenize(generated.lower())
        score = sentence_bleu(references[image_id], hypothesis, smoothing_function=smoothie)
        scores.append(score)

    average_bleu = sum(scores) / len(scores)
    print(f"Average BLEU score on test set: {average_bleu:.4f}")

# Entry point
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(CAPTION_FILE)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    vocab_stoi = checkpoint['vocab']
    encoder, decoder, _ = load_model(len(vocab_stoi), device)
    evaluate_bleu(df, encoder, decoder, vocab_stoi, device)
