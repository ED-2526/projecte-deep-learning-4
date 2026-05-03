import torch
import torch.nn as nn
import torchvision.models as models

# ============================================================
# ENCODER: CNN que llegeix la imatge i la converteix en vector
# ============================================================

class MoleculeEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])  # hasta layer3
        self.pool = nn.AdaptiveAvgPool2d((7, 7))    # (batch, 256, 7, 7)
        self.proj = nn.Linear(256, embed_dim)        # proyecta cada patch

    def forward(self, images):
        feat = self.backbone(images)                 # (batch, 256, H, W)
        feat = self.pool(feat)                       # (batch, 256, 7, 7)
        feat = feat.flatten(2).permute(0, 2, 1)     # (batch, 49, 256)
        feat = self.proj(feat)                       # (batch, 49, embed_dim)
        return feat                                  # 49 patches en vez de 1 vector


# ============================================================
# DECODER: LSTM que genera el text caràcter a caràcter
# ============================================================

class MoleculeDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim,
                            batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Atención sobre los 49 patches
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        # Inicializar hidden desde el mean de los patches
        self.img2hidden = nn.Linear(embed_dim, hidden_dim * 2)

    def forward(self, features, captions):
        # features: (batch, 49, embed_dim)
        tgt = captions[:, :-1]                          # (batch, seq_len-1)
        embeddings = self.embedding(tgt)                # (batch, seq_len-1, embed_dim)
        batch_size, seq_len, _ = embeddings.shape

        # Inicializar hidden con el mean de los patches
        mean_feat = features.mean(dim=1)                # (batch, embed_dim)
        h0 = self.img2hidden(mean_feat)
        h0 = h0.view(batch_size, 2, -1).permute(1, 0, 2).contiguous()
        c0 = torch.zeros(2, batch_size, 512).to(features.device)

        # Atención: el embedding atiende a los patches de la imagen
        context, _ = self.attention(
            query=embeddings,                           # (batch, seq_len-1, embed_dim)
            key=features,                               # (batch, 49, embed_dim)
            value=features                              # (batch, 49, embed_dim)
        )                                               # (batch, seq_len-1, embed_dim)

        # Concatenar embedding + contexto visual
        lstm_input = torch.cat([embeddings, context], dim=-1)  # (batch, seq_len-1, embed_dim*2)

        out, _ = self.lstm(lstm_input, (h0, c0))
        out = self.dropout(out)
        return self.fc(out)


# ============================================================
# MODEL COMPLET: Encoder + Decoder junts
# ============================================================

class MoleculeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = MoleculeEncoder(embed_dim)
        self.decoder = MoleculeDecoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, images, captions):
        features = self.encoder(images)                 # (batch, 49, embed_dim)
        return self.decoder(features, captions)

    def generate(self, image, idx2char, max_len=120, device='cuda'):
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(device))  # (1, 49, embed_dim)

            mean_feat = features.mean(dim=1)            # (1, embed_dim)
            h = self.decoder.img2hidden(mean_feat)
            h = h.view(1, 2, -1).permute(1, 0, 2).contiguous()  # (2, 1, hidden)
            c = torch.zeros(2, 1, 512).to(device)

            token = torch.tensor([[1]], device=device)  # <SOS>
            result = []

            for _ in range(max_len):
                emb = self.decoder.embedding(token)     # (1, 1, embed_dim)

                # Atención sobre los patches
                context, _ = self.decoder.attention(
                    query=emb,                          # (1, 1, embed_dim)
                    key=features,                       # (1, 49, embed_dim)
                    value=features
                )                                       # (1, 1, embed_dim)

                lstm_input = torch.cat([emb, context], dim=-1)  # (1, 1, embed_dim*2)
                out, (h, c) = self.decoder.lstm(lstm_input, (h, c))

                pred = self.decoder.fc(out.squeeze(1))
                next_token = pred.argmax(dim=-1)

                char = idx2char.get(next_token.item(), '')
                if char == '<EOS>':
                    break
                if char not in ('<PAD>', '<SOS>'):
                    result.append(char)

                token = next_token.unsqueeze(0)

            return ''.join(result)