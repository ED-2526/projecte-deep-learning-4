import torch
import torch.nn as nn
import torchvision.models as models

# ============================================================
# CUSTOM CNN: Red convolucional propia per a l'encoder
# ============================================================
class CustomCNN(nn.Module):
    """Red convolucional personalitzada amb stride 2, 32->512 canales.
    Entrada: imatges en escala de grisos (1 canal, 224x224)
    Sortida: vector de features (batch, output_dim)
    """
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Conv 1 ch -> 32 ch (stride 2): 224 -> 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Conv 32 ch -> 64 ch (stride 2): 112 -> 56
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Conv 64 ch -> 256 ch (stride 2): 56 -> 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Conv 256 ch -> 512 ch (stride 2): 28 -> 14
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer para ajustar la dimensión de salida
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# ============================================================
# ENCODER: CNN PRE-TRAINED que converteix la imatge en un vector
# ============================================================
class MoleculeEncoder(nn.Module):
    """Part encoder del model que s'encarrega d'extreure els feature maps de les imatges. 
    Es basa en reutilitzar els pesos de models ja entrenat, només modificant l'última capa
    perquè conincideixi amb les dimensions desitjades. S'entrena només aquesta.
    """
    def __init__(self, encoder, image_embed_dim):
        """Crea el backbone del encoder i una FC per modificar l'última capa segons embed_dim.

        Args:
            encoder (str): nom del backbone.
            embed_dim (int): dimensió del embedding que haurà de tenir les imatges.
        """
        super().__init__()
        if encoder == "resnet18": 
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        elif encoder == "resnet50": 
            # Els pesos V2 són millors
            self.backbone = models.resnet50(weights='IMAGENET1K_V2')

        elif encoder == "resnet101": 
            # Els pesos V2 són millors
            self.backbone = models.resnet101(weights='IMAGENET1K_V2')

        elif encoder == "efficientnet5": 
            self.backbone = models.efficientnet_b5(weights='IMAGENET1K_V1')

        elif encoder == "conv":
            self.backbone = CustomCNN(image_embed_dim)
        
        if encoder in ["resnet18", "resnet50", "resnet101"]:
            # Congela TOT primer
            for param in self.backbone.parameters():
                param.requires_grad_(False)

            # Descongela l'últim bloc convolucional (layer4 a ResNet)
            for param in self.backbone.layer4.parameters():
                param.requires_grad_(True)

            # Descongela la capa fc final (sempre entrenable)
            self.backbone.fc = nn.Linear(
                self.backbone.fc.in_features, image_embed_dim
            )
            # backbone.fc ja té requires_grad=True per defecte

        elif encoder == "efficientnet5":
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features, image_embed_dim)
            )
        
    def forward(self, images):
        features = self.backbone(images)                    # ==> (batch, image_embed_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)         # ==> (batch, image_embed_dim)
        return features

# ============================================================
# DECODER: LSTM que genera text segons caràcters i feature maps
# ============================================================
class MoleculeDecoder(nn.Module):
    """Part decoder del model que s'encarrega de generar text segons informació 
    passada. Es basa en un LSTM. S'entrena totalment.
    """
    def __init__(self, vocab_size, caption_embed_dim, image_embed_dim, hidden_dim, dropout, num_layers):
        """Crea el LSTM, el DropOut i el Embedding per tractar els vectors d'imatges i 
        els vectors de captions. 

        Args:
            vocab_size (int): número de caràcters únics.
            caption_embed_dim (int): dimensió del embedding dels captions.
            image_embed_dim (int): dimensió del embedding de les imatges.
            hidden_dim (int): dimensió hidden del LSTM.
            dropout (float): percentatge de dropout.
            num_layers (int): número de LSTM per ajuntar.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, caption_embed_dim)
        self.lstm = nn.LSTM(caption_embed_dim + image_embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # S'encarrega de passar de hidden_dim a vocab_size per aconseguir els diferents logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Transforma la dimensió del vector de la imatge perquè coincideixi amb hidden (en el cas que no siguin iguals)
        self.img2hidden = nn.Linear(image_embed_dim, hidden_dim)
        
    def forward(self, seq, h, c, features=None):  
        embedding = self.embedding(seq)                     #  ==> (batch, seq_len, caption_embed_dim)
        if features is not None:
            f = features.unsqueeze(1).expand(-1, embedding.size(1), -1)
            lstm_input = torch.cat([embedding, f], dim=-1)  #  ==> (batch, seq_len, caption_embed_dim + image_embed_dim)
        else:
            lstm_input = embedding
        out, (h, c) = self.lstm(lstm_input, (h, c))         #  ==> (batch, seq_len, hidden_dim)
        out = self.dropout(out)                             #  ==> (batch, seq_len, hidden_dim)
        out = self.fc(out)                                  #  ==> (batch, seq_len, vocab_size)
        return out, h, c                           

    def init_state(self, features): 
        h = self.img2hidden(features).unsqueeze(0)  # ==> (1, batch, hidden_dim)
        c = torch.zeros(self.num_layers, features.size(0), self.hidden_dim).to(features.device)   # ==> (1, batch, hidden_dim)
        return h, c

    
# ============================================================
# MODEL COMPLET: Encoder + Decoder junts
# ============================================================
class MoleculeModel(nn.Module):
    """Encoder i Decoder junt que s'encarrega de generar text segons una imatge
    i una caption donada. 
    """
    def __init__(self, encoder, image_embed_dim, caption_embed_dim, hidden_dim,
                 vocab_size, max_len, diccionaris, dropout, num_layers=1):
        """Crear les dues parts del model: Encoder i Decoder

        Args:
            encoder (str): nom del encoder.
            image_embed_dim (_type_): dimensió del embedding de les imatges.
            caption_embed_dim (_type_): dimensió del embedding de les imatges.
            hidden_dim (_type_): dimensió hidden del LSTM.
            vocab_size (_type_): número de caràcters únics.
            max_len (_type_): màxim longitud de SMILES trobada al dataset.
            diccionaris (_type_): diccionaris per covertir de char a idx i a l'inrevés.
            dropout (_type_): percentatge de dropout.
            num_layers (_type_): número de LSTM per ajuntar.
        """
        super().__init__()

        self.max_len = max_len
        self.char2idx = diccionaris[0]
        self.idx2char = diccionaris[1]

        self.encoder = MoleculeEncoder(encoder, image_embed_dim)
        self.decoder = MoleculeDecoder(vocab_size, caption_embed_dim, image_embed_dim,
                                       hidden_dim, dropout, num_layers)

        # Llista de paràmetres entrenables per passar al 
        self.params_train = [param for param in self.parameters() if param.requires_grad]
        

    def forward(self, image, seq):
        """Dona les prediccions d'una imatge i una seqüència.

        Args:
            image (tensor): imatge per calcular el state.
            seq (tensor): caption per predir.

        Returns:
            out (tensor): predicció.
            h (tensor): hidden state últims.
            c (tensor): cell state últims.
        """
        features = self.encoder(image)                      # ==> (batch, image_embed_dim)
        h, c = self.decoder.init_state(features)
        out, h, c = self.decoder(seq, h, c, features=features)
        return out, h, c

    def predict(self, seq, h, c, features=None): 
        """Calcula una predicció segons un state i una seqüència. 

        Args:
            seq (tensor): sequència a predir. 
            h (tensor): hidden state.
            c (tensor): cell state.
            features (tensor, optional): image features to inject at each step.

        Returns:
            out (tensor): predicció.
            h (tensor): hidden state últim.
            c (tensor): cell state últim.
        """
        out, h, c = self.decoder(seq, h, c, features=features)
        return out, h, c

    def generate_prediction(self, image, device='cuda'):
        """Genera la predicció d'una imatge fins <EOS>.

        Args:
            image (tensor): imatge de la molècula.
            device (str, optional): dispositiu on calcular les operacions.

        Returns:
            text (str): SMILE de la molècula.
        """
        self.eval()

        image = image.unsqueeze(0)
        features = self.encoder(image)
        h, c = self.decoder.init_state(features)

        text = ""
        token = torch.tensor([[1]]).to(device)  # 1 = <SOS>
        
        with torch.no_grad():
            for _ in range(self.max_len): 
                pred, h, c = self.predict(token, h, c, features=features)
                token = torch.argmax(pred, dim=2)
                char = self.idx2char.get(token.item(), '')

                if char == "<EOS>": 
                    break
                text += char

        return text
    
    def generate_smiles(self, caption): 
        text = ""

        for token in caption[1:]: 
            char = self.idx2char.get(token.item(), '') 

            if char == "<EOS>":
                break
            text += char

        return text

    def generate_beam(self, image, device='cuda', beam_size=3):
        """Genera SMILES amb Beam Search."""
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            features = self.encoder(image)
            h, c = self.decoder.init_state(features)

            # Cada beam: (tokens_generats, log_prob_acumulada, h, c)
            beams = [([1], 0.0, h, c)]  # 1 = <SOS>

            for _ in range(self.max_len):
                candidates = []

                for tokens, score, h_beam, c_beam in beams:
                    last_token = torch.tensor([[tokens[-1]]]).to(device)
                    out, h_new, c_new = self.predict(last_token, h_beam, c_beam, features=features)

                    # out: (1, 1, vocab_size)
                    log_probs = torch.log_softmax(out[0, 0], dim=0)
                    top_probs, top_tokens = torch.topk(log_probs, beam_size)

                    for prob, tok in zip(top_probs, top_tokens):
                        new_tokens = tokens + [tok.item()]
                        new_score = score + prob.item()
                        candidates.append((new_tokens, new_score, h_new, c_new))

                # Ordena per score i queda amb els millors beam_size
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]

                # Para si tots han generat <EOS>
                if all(b[0][-1] == 2 for b in beams):  # 2 = <EOS>
                    break

            # Retorna el millor beam
            best_tokens = beams[0][0]
            text = ''
            for tok in best_tokens[1:]:  # salta <SOS>
                char = self.idx2char.get(tok, '')
                if char == '<EOS>':
                    break
                if char not in ['<PAD>', '<SOS>']:
                    text += char
            return text
        