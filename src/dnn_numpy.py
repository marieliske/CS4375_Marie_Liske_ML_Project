import torch
import torch.nn as nn
from torchvision import models

# This file has the model definition - encoder (ResNet-50) and decoder (LSTM)
# The LSTM gates are the core algorithm/technique that was coded from scratch

class Encoder(nn.Module):
    # uses pretrained ResNet-50 to turn an image into a feature vector
    # I removed the last fc layer and added my own projection layer
    def __init__(self, feature_dim=2048, freeze_backbone=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_proj = nn.Linear(2048, feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def extract_raw(self, images):
        # Run just the backbone (no projection), used for caching features before training
        return self.backbone(images).flatten(1)

    def project(self, raw_features):
        # Apply projection + layernorm to cached features
        return self.feature_norm(self.feature_proj(raw_features))

    def forward(self, images):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone(images).flatten(1)
        features = self.feature_proj(features)
        return self.feature_norm(features)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Concatenate word embedding + image feature at every step
        self.x2h = nn.Linear(embed_dim + encoder_dim, 4 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def step(self, x, h_prev, c_prev):
        # LSTM update - 4 gates packed into one linear layer then split
        # i_t = input gate (what new info to write)
        # f_t = forget gate (what to erase from cell state)
        # g_t = cell gate (candidate new content)
        # o_t = output gate (what to read out)
        gates = self.x2h(x) + self.h2h(h_prev)
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
        _, seq_len, _ = embeddings.shape

        # Initialize hidden and cell state from image features
        h = torch.tanh(self.init_h(image_features))
        c = torch.tanh(self.init_c(image_features))

        # Tile image features across the whole sequence so we can concat at each step
        img_feat = image_features.unsqueeze(1).expand(-1, seq_len, -1)

        step_logits = []
        for t in range(seq_len):
            x = torch.cat([embeddings[:, t, :], img_feat[:, t, :]], dim=-1)
            h, c = self.step(x, h, c)
            logits = self.output(self.dropout(h))
            step_logits.append(logits)

        logits = torch.stack(step_logits, dim=1)
        return logits


class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_dim=512, freeze_encoder=True):
        super().__init__()
        self.encoder = Encoder(feature_dim=encoder_dim, freeze_backbone=freeze_encoder)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
        )

    def forward(self, images, captions_in, raw_features=None):
        if raw_features is not None:
            image_features = self.encoder.project(raw_features)
        else:
            image_features = self.encoder(images)
        return self.decoder(image_features, captions_in)

    @torch.no_grad()
    def generate(self, image, start_idx, end_idx, max_len=25, raw_feature=None, beam_size=1):
        self.eval()
        if raw_feature is not None:
            features = self.encoder.project(raw_feature.unsqueeze(0) if raw_feature.dim() == 1 else raw_feature)
            device = raw_feature.device
        else:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = self.encoder(image)
            device = image.device

        h0 = torch.tanh(self.decoder.init_h(features))
        c0 = torch.tanh(self.decoder.init_c(features))

        if beam_size <= 1:
            # Greedy decoding: pick the highest probability word at each step
            h, c = h0, c0
            token = torch.tensor([[start_idx]], device=device)
            feat = features.squeeze(0)
            generated = [start_idx]
            for _ in range(max_len):
                emb = self.decoder.embedding(token).squeeze(1)
                x = torch.cat([emb, feat.unsqueeze(0)], dim=-1)
                h, c = self.decoder.step(x, h, c)
                logits = self.decoder.output(h)
                next_id = torch.argmax(logits, dim=-1)
                next_val = int(next_id.item())
                generated.append(next_val)
                token = next_id.unsqueeze(1)
                if next_val == end_idx:
                    break
            return generated

        # Beam search - keep the top beam_size candidates at each step
        # each beam is (cumulative_log_prob, token_id_list, h, c)
        feat = features.squeeze(0)
        beams = [(0.0, [start_idx], h0, c0)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break
            candidates = []
            for score, seq, h, c in beams:
                if seq[-1] == end_idx:
                    completed.append((score, seq))
                    continue
                token = torch.tensor([[seq[-1]]], device=device)
                emb = self.decoder.embedding(token).squeeze(1)
                x = torch.cat([emb, feat.unsqueeze(0)], dim=-1)
                h_new, c_new = self.decoder.step(x, h, c)
                log_probs = torch.log_softmax(self.decoder.output(h_new), dim=-1).squeeze(0)
                top_probs, top_ids = log_probs.topk(beam_size)
                for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                    candidates.append((score + prob, seq + [idx], h_new, c_new))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

        for score, seq, _, __ in beams:
            completed.append((score, seq))

        if not completed:
            return [start_idx, end_idx]

        # Divide by length so shorter sequences dont get unfairly high scores
        best = max(completed, key=lambda x: x[0] / max(len(x[1]) - 1, 1))
        return best[1]
