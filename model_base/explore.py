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

# Mostra del funcionament del LSTM
x = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]], 
                  [[1, 2, 3, 4], [1, 2, 3, 4]]]).to(torch.float32) 

x_prima = torch.tensor([[[1, 2, 3, 4]]]).to(torch.float32) 

h = torch.tensor([[[0, 0, 0, 0, 0, 0]]]).to(torch.float32)
c = torch.tensor([[[0, 0, 0, 0, 0, 0]]]).to(torch.float32)
lstm = nn.LSTM(input_size=4, hidden_size=6, batch_first=True)
pred, _ = lstm(x)
pred1, (h, c) = lstm(x_prima, (h, c))
pred2, (h, c) = lstm(x_prima, (h, c))
print(pred)
print(pred1)
print(pred2)

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
