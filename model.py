import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        att1 = self.encoder_att(features)         # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        att = torch.tanh(att1 + att2)
        alpha = torch.softmax(self.full_att(att), dim=1)  # (batch_size, num_pixels, 1)
        context = (features * alpha).sum(dim=1)  # (batch_size, encoder_dim)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim=2048):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        embeddings = self.embedding(captions[:, :-1])  # skip <END>
        h, c = (torch.zeros(batch_size, self.lstm.hidden_size).to(features.device),
               torch.zeros(batch_size, self.lstm.hidden_size).to(features.device))

        outputs = []
        for t in range(embeddings.size(1)):
            context, _ = self.attention(features, h)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(h)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)  # (B, 2048, H/32, W/32)
            features = self.pool(features)    # (B, 2048, encoded_image_size, encoded_image_size)
            features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # (B, num_pixels, 2048)
        return features
