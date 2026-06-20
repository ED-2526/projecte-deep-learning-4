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

torch.manual_seed(42)

from models import MoleculeModel

# unfreeze = 3
# model = MoleculeModel("resnet50", 256, 256, 512, unfreeze, vocab_size=54, max_len=367, diccionaris=(dict(), dict()), dropout=0.8, num_layers=1)


# x = torch.rand(1, 3, 224, 224)
# target = torch.tensor([860])
# # y = model(x)
# # y_prima = torch.argmax(y, dim=1)

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.conv1.parameters())
# optimizer.add_param_group({'params': model.bn1.parameters()})
# optimizer.add_param_group({'params': model.layer1.parameters()})
# optimizer.add_param_group({'params': model.layer2.parameters()})
# optimizer.add_param_group({'params': model.layer3.parameters()})
# optimizer.add_param_group({'params': model.layer4.parameters()})
# optimizer.add_param_group({'params': model.fc.parameters()})

# # optimizer = torch.optim.Adam(model.parameters())

# for i in range(5): 
#     y = model(x)
#     loss = criterion(y, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(loss)
# model = torch.load(f"{a}.pth")

def test(a): 
    print("TEST")
    return a

test(20)

# model = models.resnet18()
# for i, (name, layer) in enumerate(model.named_children()): 
#     print(i, name)
# a = "states/resnet50+unfreeze"
# # for param in model.parameters():
# #     param.requires_grad_(False)
# torch.save(model.state_dict(), f"{a}.pth")

# summary(model)
# print(model)

# for param in model.parameters(): 
#     param.requires_grad_(False)

# chil = [c for c in model.children()] 
# for param in chil[0].parameters(): 
#     param.requires_grad_(True)

# summary(model)

# until = 1

# if until != 0: 
#     print(4)

#     if until <= 3:
#         print(3)

#         if until <= 2: 
#             print(2)

#             if until <= 1: 
#                 print(1)
        
    

# until = 3
# for a in range(4, until-1, -1): 
#     print(a)

# a = [1, 3, 2, 4]
# a.sort()
# print(a)
# Mostra del funcionament del LSTM
# x = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]], 
#                   [[1, 2, 3, 4], [1, 2, 3, 4]]]).to(torch.float32) 

# x_prima = torch.tensor([[[1, 2, 3, 4]]]).to(torch.float32) 

# h = torch.tensor([[[0, 0, 0, 0, 0, 0]]]).to(torch.float32)
# c = torch.tensor([[[0, 0, 0, 0, 0, 0]]]).to(torch.float32)
# lstm = nn.LSTM(input_size=4, hidden_size=6, batch_first=True)
# pred, _ = lstm(x)
# pred1, (h, c) = lstm(x_prima, (h, c))
# pred2, (h, c) = lstm(x_prima, (h, c))
# print(pred)
# print(pred1)
# print(pred2)

# Prova disminució tf_ration
# tf_ratio = 1
# for epoch in range(150):
#     tf_ratio = 1 - np.floor(epoch/14)*0.1
#     print(epoch, tf_ratio)

# print(torch.argmax(x, dim=2))
# t = 0
# exact = torch.equal(x, y)
# t+=exact
# l = [1, 2, 3, 4]
# d = {1: 'hola', 2: 'dos', 3: 'tres'}
# smiles = ''.join(d.get(t.item(), '') 
#                  if d.get(t.item(), '') not in [0] 
#                  else None
#                  for t in x
#             )

# print("Carregant dataset...")
# dataset = load_dataset("docling-project/USPTO-30K")
# clean = dataset["clean"]

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




