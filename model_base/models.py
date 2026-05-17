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
        elif encoder == "conv":
            self.backbone = CustomCNN(image_embed_dim)
        
        # Es congelen totes les capas amb Requires_grad=False
        if encoder != "conv":
            for param in self.backbone.parameters(): 
                param.requires_grad_(False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, image_embed_dim)
        
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
        self.lstm = nn.LSTM(caption_embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # S'encarrega de passar de hidden_dim a vocab_size per aconseguir els diferents logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Transforma la dimensió del vector de la imatge perquè coincideixi amb hidden (en el cas que no siguin iguals)
        self.img2hidden = nn.Linear(image_embed_dim, hidden_dim)
        
    def forward(self, seq, h, c):  
        embedding = self.embedding(seq)                     #  ==> (batch, seq_len, caption_embed_dim)
        out, (h, c) = self.lstm(embedding, (h, c))          #  ==> (batch, seq_len, hidden_dim)
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
        out, h, c = self.decoder(seq, h, c)                  # ==> (batch, seq_len, vocab_size), (1, batch, hidden_dim)
        return out, h, c

    def predict(self, seq, h, c): 
        """Calcula una predicció segons un state i una seqüència. 

        Args:
            seq (tensor): sequència a predir. 
            h (tensor): hidden state.
            c (tensor): cell state.

        Returns:
            out (tensor): predicció.
            h (tensor): hidden state últim.
            c (tensor): cell state últim.
        """
        out, h, c = self.decoder(seq, h, c)
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
                pred, h, c = self.predict(token, h, c)
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
        