import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """
    Pretrained ResNet-50 feature extractor.
    The encoder is frozen by default for transfer learning.
    """

    def __init__(self, feature_dim=2048, freeze_backbone=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_proj = nn.Linear(2048, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            feats = self.backbone(images).flatten(1)
        feats = self.feature_proj(feats)
        return self.feature_norm(feats)


class ManualLSTMDecoder(nn.Module):
    """
    LSTM decoder written explicitly with gate equations.
    This keeps the main sequence-generation algorithm in project code.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.x2h = nn.Linear(embed_dim, 4 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def step(self, x_t, h_prev, c_prev):
        gates = self.x2h(x_t) + self.h2h(h_prev)
        i_t, f_t, g_t, o_t = torch.chunk(gates, chunks=4, dim=-1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def forward(self, image_features, captions_in):
        embeddings = self.embedding(captions_in)
        batch_size, seq_len, _ = embeddings.shape

        h_t = torch.tanh(self.init_h(image_features))
        c_t = torch.tanh(self.init_c(image_features))

        logits_steps = []
        for t in range(seq_len):
            x_t = embeddings[:, t, :]
            h_t, c_t = self.step(x_t, h_t, c_t)
            logits_t = self.output(self.dropout(h_t))
            logits_steps.append(logits_t)

        logits = torch.stack(logits_steps, dim=1)
        return logits


class CaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        encoder_dim=512,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = EncoderCNN(feature_dim=encoder_dim, freeze_backbone=freeze_encoder)
        self.decoder = ManualLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
        )

    def forward(self, images, captions_in):
        image_features = self.encoder(images)
        return self.decoder(image_features, captions_in)

    @torch.no_grad()
    def generate(self, image, start_idx, end_idx, max_len=25):
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        features = self.encoder(image)
        h_t = torch.tanh(self.decoder.init_h(features))
        c_t = torch.tanh(self.decoder.init_c(features))

        token = torch.tensor([[start_idx]], device=image.device)
        generated = [start_idx]

        for _ in range(max_len):
            emb = self.decoder.embedding(token).squeeze(1)
            h_t, c_t = self.decoder.step(emb, h_t, c_t)
            logits = self.decoder.output(h_t)
            next_token = torch.argmax(logits, dim=-1)
            next_idx = int(next_token.item())
            generated.append(next_idx)
            token = next_token.unsqueeze(1)
            if next_idx == end_idx:
                break

        return generated
