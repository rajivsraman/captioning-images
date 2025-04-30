import os
import streamlit as st
import torch
import torch.nn as nn
import warnings
from PIL import Image
import requests
from io import BytesIO
from model import DecoderWithAttention
from dataset import get_transforms
import nltk

nltk.download('punkt')

# Suppress warnings and printing
warnings.filterwarnings("ignore")
torch.set_printoptions(profile="default")

# Constants
GCS_MODEL_URL = "https://storage.googleapis.com/imgcap-deep-model/best_caption_model.pth"
RESNET_URL = "https://storage.googleapis.com/imgcap-deep-model/resnet50.pth"
MODEL_LOCAL_PATH = "best_caption_model.pth"
RESNET_LOCAL_PATH = "resnet50.pth"
MAX_LEN = 20

# Download model checkpoints if not already present
def download_model_files():
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.info("Downloading captioning model...")
        response = requests.get(GCS_MODEL_URL)
        with open(MODEL_LOCAL_PATH, 'wb') as f:
            f.write(response.content)

    if not os.path.exists(RESNET_LOCAL_PATH):
        st.info("Downloading ResNet-50 backbone...")
        response = requests.get(RESNET_URL)
        with open(RESNET_LOCAL_PATH, 'wb') as f:
            f.write(response.content)

# Modified EncoderCNN class using local ResNet weights
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torch.hub.load('pytorch/vision', 'resnet50', weights=None)
        resnet.load_state_dict(torch.load(RESNET_LOCAL_PATH, map_location=torch.device('cpu'), weights_only=False))
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)  # [B, enc_image_size, enc_image_size, 2048]
        return out

# Load model and vocab
def load_model(device):
    encoder = EncoderCNN().to(device)
    checkpoint = torch.load(MODEL_LOCAL_PATH, map_location=device)
    decoder = DecoderWithAttention(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(checkpoint['vocab']),
        attention_dim=256
    ).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint['vocab']

# Generate caption
def generate_caption(image, encoder, decoder, vocab, device):
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)
    vocab_inv = {idx: tok for tok, idx in vocab.items()}

    with torch.no_grad():
        features = encoder(image_tensor)
        word = torch.tensor([vocab['<START>']]).to(device)
        h, c = torch.zeros(1, 512).to(device), torch.zeros(1, 512).to(device)
        caption = []

        for _ in range(MAX_LEN):
            emb = decoder.embedding(word).squeeze(1)
            context, _ = decoder.attention(features, h)
            lstm_input = torch.cat([emb, context], dim=1)
            h, c = decoder.lstm(lstm_input, (h, c))
            output = decoder.fc(h)
            word = output.argmax(1).unsqueeze(1)
            pred_word = vocab_inv[word.item()]
            if pred_word == "<END>":
                break
            caption.append(pred_word)

    return ' '.join(caption)

# Streamlit app
def main():
    st.title("Image Captioning Module")
    st.write("Upload an image, and the model will generate a caption!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_model_files()

    # Load model once
    model_container = {}
    def init_models():
        encoder, decoder, vocab = load_model(device)
        model_container['encoder'] = encoder
        model_container['decoder'] = decoder
        model_container['vocab'] = vocab
    init_models()

    st.header("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        caption = generate_caption(
            image,
            model_container['encoder'],
            model_container['decoder'],
            model_container['vocab'],
            device
        )
        st.image(image, caption=f"**Predicted Caption:** {caption}", use_container_width=True)

if __name__ == "__main__":
    main()
