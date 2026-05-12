from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch
from torchinfo import summary
import torchvision.models as models
from torchvision import transforms 
import torch.optim as optim

# resnet = models.resnet18(weights='IMAGENET1K_V1')
# backbone = nn.Sequential(*list(resnet.children())[:-1])

# for param in backbone.parameters(): 
#     param.requires_grad_(False)

# print(resnet)
# summary(resnet)

# print(backbone)
# summary(backbone)


import torch
import torch.nn as nn
import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ============================================================
# ENCODER: CNN que llegeix la imatge i la converteix en vector
# ============================================================
class MoleculeEncoder(nn.Module):
    def __init__(self, encoder, embed_dim=256):
        super().__init__()
        # Usem ResNet18 preentrenat a ImageNet
        # (ja sap reconèixer formes, vores, textures)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        # Traiem l'última capa (classificació de 1000 classes d'ImageNet)
        # Ens quedem tot menys l'últim fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Posar Requires_grad=False en totes les capes del backbone (Última)
        for param in self.backbone.parameters(): 
            param.requires_grad_(False)
        
        # Afegim una capa per reduir de 512 (última capa resnet18) a embed_dim 
        # (aquesta té Requires_grad=True)
        self.fc = nn.Linear(512, embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, images):
        features = self.backbone(images)                # (batch, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)     # (batch, 512)
        features = self.relu(self.fc(features))         # (batch, embed_dim=256)
        return features

# ============================================================
# DECODER: LSTM que genera el text caràcter a caràcter
# ============================================================
class MoleculeDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

        # Projecta el context de la imatge a hidden_dim per inicialitzar la LSTM
        self.img2hidden = nn.Linear(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, features, captions):  
        # Usem la imatge com a h0 de la LSTM (estat inicial)
        batch_size = features.size(0)
        h0 = self.img2hidden(features).unsqueeze(0)    # (1, batch, hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(features.device)

        embeddings = self.embedding(captions[:, :-1])  # (batch, max_len-1, embed_dim)
        out, _ = self.lstm(embeddings, (h0, c0))       # (batch, max_len-1, hidden_dim)
        out = self.dropout(out)
        return self.fc(out)                             # (batch, max_len-1, vocab_size)

# ============================================================
# MODEL COMPLET: Encoder + Decoder junts
# ============================================================
class MoleculeModel(nn.Module):
    def __init__(self, encoder, vocab_size, max_len, idx2char, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.max_len = max_len
        self.idx2char = idx2char
        self.encoder = MoleculeEncoder(encoder, embed_dim)
        self.decoder = MoleculeDecoder(vocab_size, embed_dim, hidden_dim)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
    
    def generate(self, image, idx2char, device='cuda'):
        """Genera text a partir d'una imatge (inferència)"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(device))
            
            # Comencem amb el token <SOS>
            token = torch.tensor([[1]]).to(device)  # 1 = <SOS>
            h = self.decoder.img2hidden(features).unsqueeze(0)
            c = torch.zeros(1, 1, self.decoder.hidden_dim).to(device)
            
            result = []
            img_injected = False
            
            for _ in range(self.max_len+2):
                emb = self.decoder.embedding(token)  # (1, 1, embed_dim)
                
                if not img_injected:
                    emb = features.unsqueeze(1)
                    img_injected = True
                
                out, (h, c) = self.decoder.lstm(emb, (h, c))
                pred = self.decoder.fc(out.squeeze(1))
                next_token = pred.argmax(dim=-1)
                
                char = self.idx2char.get(next_token.item(), '') #passa de token a caracter
                if char == '<EOS>':
                    break

                if char not in ['<PAD>', '<SOS>']: #evita tokens inutils
                    result.append(char) 

                    
                token = next_token.unsqueeze(0)
            
            return ''.join(result)
        
# model = MoleculeModel(
#             encoder="Resnet18", 
#             vocab_size=54,
#             max_len=120,
#             idx2char=dict(),
#             embed_dim=256,
#             hidden_dim=512
#         ) #Puja model a GPU/CPU


model = models.resnet50(weights="IMAGENET1K_V1")
summary(model)
print(model)

# # print(f"Descripció de capes del model:\n{model}")
# summary(model)

# params = [param for param in model.parameters() if param.requires_grad]
# optimizer_ft = optim.Adam(params, lr=0.001)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))


# class MoleculeEncoder(nn.Module):
#     def __init__(self, embed_dim=256):
#         super().__init__()
        
#         # Usem ResNet18 preentrenat a ImageNet
#         # (ja sap reconèixer formes, vores, textures)
#         resnet = models.resnet18(weights='IMAGENET1K_V1')
        
#         # Traiem l'última capa (classificació de 1000 classes d'ImageNet)
#         # Ens quedem tot menys l'últim fc
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])

#         # Posar Requires_grad=False en totes les capes del backbone (Última)
#         for param in self.backbone.parameters(): 
#             param.requires_grad_(False)
        
#         # Afegim una capa per reduir de 512 (última capa backbone) a embed_dim 
#         # (aquesta té Requires_grad=True)
#         self.fc = nn.Linear(512, embed_dim)
#         self.relu = nn.ReLU()
        
#     def forward(self, images):
#         features = self.backbone(images)
#         print(features.shape)
#         features = features.squeeze(-1).squeeze(-1)  # (batch, 512)
#         print(features.shape)
#         features = self.relu(self.fc(features))       # (batch, embed_dim=256)
#         print(features.shape)
#         return features
    
# model = MoleculeEncoder()

# x = torch.rand((5, 3, 32, 32))
# x = torch.arange(5)
# print(x)
# print(x[1:])
# print(x.shape)

# y = model(x)


# print("Carregant dataset...")
# dataset = load_dataset("docling-project/USPTO-30K")

# clean = dataset["clean"]

# transform = transforms.Compose([
#             transforms.Resize((244, 244)),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.ToTensor()
#         ])

# for i in range(2):
#     example = clean[i]

#     im = example["image"]
#     im = transform(im)

#     im_array = np.asarray(im)
#     print(im_array.shape)
#     # print(np.unique(im_array))
#     print(np.all(im_array[:, :, 0] == im_array[:, :, 1])&
#           np.all(im_array[:, :, 1] == im_array[:, :, 2]))

# # im.save('exemple_molecula.png')


# tr_im = transform(im)

# print(np.asarray(tr_im).shape)
# print(np.unique(tr_im))

# print("Splits disponibles:", dataset)
# print("Clean:", len(dataset['clean']))
# print("Abbreviated:", len(dataset['abbreviated']))
# print("Large:", len(dataset['large']))

# exemple = dataset['clean'][0]
# print("\nClaus de cada exemple:", exemple.keys())
# print("\nNom del fitxer:", exemple['filename'])
# print("\nText MolFile (primers 300 caràcters):")
# print(exemple['mol'][:300])

# exemple['image'].save('exemple_molecula.png')
# print("\nImatge guardada com exemple_molecula.png")

# imatges_mides = [dataset['clean'][i]['image'].size for i in range(10)]
# print("\nMides de les primeres 10 imatges:", imatges_mides)

# longituds = [len(dataset['clean'][i]['mol']) for i in range(100)]
# print(f"\nLongitud text MolFile - Mínim: {min(longituds)}, Màxim: {max(longituds)}, Mitjana: {sum(longituds)//len(longituds)}")
