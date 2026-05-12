import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MoleculeEncoder(nn.Module):
    def __init__(self, embed_dim=256, freeze_resnet=True, backbone_name='resnet18'):
        super().__init__()

        # Selecciona el backbone i la dimensió de sortida
        if backbone_name == 'resnet18':
            resnet = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1')
            feature_dim = 2048

        # Adapta el primer canal per imatges en grisos (1 canal en lloc de 3)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)

        # Keepem el feature map espacial 7x7 (traiem els 2 últims mòduls)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Congela ResNet: no s'actualitzen els pesos preentrenats
        if freeze_resnet:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  {backbone_name} congelada (freeze_resnet=True)")
        else:
            print(f"  {backbone_name} NO congelada (finetuning complet)")

        # Projecció: feature_dim (512 o 2048) → embed_dim (256)
        self.fc = nn.Linear(feature_dim, embed_dim)
        
    def forward(self, images):
        # images: (batch, 1, 224, 224)
        features = self.backbone(images)      # (batch, 512, 7, 7)
        batch, C, H, W = features.shape
        
        # Aplanem les posicions espacials
        features = features.view(batch, C, H*W)  # (batch, 512, 49)
        features = features.permute(0, 2, 1)      # (batch, 49, 512)
        features = self.fc(features)              # (batch, 49, embed_dim)
        # Ara tenim 49 vectors (un per cada regió de la imatge 7x7)
        return features


class AttentionLayer(nn.Module):
    """Calcula quines regions de la imatge mirar en cada pas"""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.W_img = nn.Linear(embed_dim, embed_dim)
        self.W_hid = nn.Linear(hidden_dim, embed_dim)
        self.v = nn.Linear(embed_dim, 1)
        
    def forward(self, img_features, hidden):
        # img_features: (batch, 49, embed_dim) — les 49 regions
        # hidden: (batch, hidden_dim) — estat actual de la LSTM
        
        # Projectem la imatge i el hidden state al mateix espai
        img_proj = self.W_img(img_features)           # (batch, 49, embed_dim)
        hid_proj = self.W_hid(hidden).unsqueeze(1)    # (batch, 1, embed_dim)
        
        # Calculem scores d'atenció (quina regió és important ara)
        scores = self.v(torch.tanh(img_proj + hid_proj))  # (batch, 49, 1)
        weights = F.softmax(scores, dim=1)                 # (batch, 49, 1)
        
        # Combinem les regions pesades per l'atenció
        context = (weights * img_features).sum(dim=1)      # (batch, embed_dim)
        return context, weights


class MoleculeDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # La LSTM rep: embedding + context de l'atenció
        self.lstm = nn.LSTMCell(embed_dim + embed_dim, hidden_dim)
        
        self.attention = AttentionLayer(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
        # Inicialitza h0 i c0 a partir de la mitjana de les features
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)

  
        
    def forward(self, img_features, captions):
        # img_features: (batch, 49, embed_dim)
        # captions: (batch, seq_len)
        
        batch_size = img_features.size(0)
        seq_len = captions.size(1) - 1  # traiem l'últim token
        
        # Inicialitzem h i c a partir de la mitjana de la imatge
        mean_features = img_features.mean(dim=1)  # (batch, embed_dim)
        h = torch.tanh(self.init_h(mean_features))  # (batch, hidden_dim)
        c = torch.tanh(self.init_c(mean_features))  # (batch, hidden_dim)
        
        # Embeddings dels caràcters d'entrada (sense l'últim)
        embeddings = self.embedding(captions[:, :-1])  # (batch, seq_len, embed_dim)
        
        outputs = []
        for t in range(seq_len):
            # Atenció: quin tros de la imatge mirem ara?
            context, _ = self.attention(img_features, h)  # (batch, embed_dim)
            
            # Input de la LSTM: caràcter actual + context visual
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            
            # Un pas de la LSTM
            h, c = self.lstm(lstm_input, (h, c))
            h = self.dropout(h)
            
            # Predicció del proper caràcter
            out = self.fc(h)  # (batch, vocab_size)
            outputs.append(out)
        
        # (seq_len, batch, vocab_size) → (batch, seq_len, vocab_size)
        return torch.stack(outputs, dim=1)


class MoleculeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, freeze_resnet=True, backbone_name='resnet18'):
        super().__init__()
        self.encoder = MoleculeEncoder(embed_dim, freeze_resnet=freeze_resnet,backbone_name=backbone_name)
        self.decoder = MoleculeDecoder(vocab_size, embed_dim, hidden_dim)
        
    def forward(self, images, captions):
        img_features = self.encoder(images)  # (batch, 49, embed_dim)
        return self.decoder(img_features, captions)
    
    def generate(self, image, idx2char, max_len=200, device='cuda'):
        """Genera SMILES a partir d'una imatge"""
        self.eval()
        with torch.no_grad():
            img_features = self.encoder(image.unsqueeze(0).to(device))
            
            # Inicialitzem h i c
            mean_features = img_features.mean(dim=1)
            h = torch.tanh(self.decoder.init_h(mean_features))
            c = torch.tanh(self.decoder.init_c(mean_features))
            
            # Comencem amb <SOS>
            token = torch.tensor([1]).to(device)  # 1 = <SOS>
            result = []
            
            for _ in range(max_len):
                emb = self.decoder.embedding(token)        # (1, embed_dim)
                context, _ = self.decoder.attention(img_features, h)
                
                lstm_input = torch.cat([emb, context], dim=1)
                h, c = self.decoder.lstm(lstm_input, (h, c))
                
                pred = self.decoder.fc(h)
                next_token = pred.argmax(dim=-1)
                
                char = idx2char.get(next_token.item(), '')
                if char == '<EOS>':
                    break
                if char not in ['<PAD>', '<SOS>']:
                    result.append(char)
                
                token = next_token
            
            return ''.join(result)