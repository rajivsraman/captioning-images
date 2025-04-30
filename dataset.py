import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import torchvision.transforms as transforms

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in word_tokenize(sentence.lower()):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in word_tokenize(text.lower())]

class FlickrDataset(Dataset):
    def __init__(self, dataframe, vocab, image_dir, transform=None):
        self.df = dataframe
        self.vocab = vocab
        self.image_dir = image_dir
        self.transform = transform
        self.images = self.df["image"].values
        self.captions = self.df["caption"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.images[index]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<START>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numerical_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
