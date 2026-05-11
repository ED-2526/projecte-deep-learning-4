import numpy as np
import random
import wandb #Weights & Biases

import torch 
from torchinfo import summary
import torch.nn as nn #Submòdul de PyTorch per xarxes neuronals.

from models.models import MoleculeModel
from train import train
from utils.utils import make_loaders

# Reproduïbilitat
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usant device: {device}")


def model_pipeline(cfg):
    with wandb.init(project="molecule-recognition", config=cfg):
        config = wandb.config

        # 1. Carregar dades
        print("\n--- Carregant dades ---")
        train_loader, val_loader, vocab_size, idx2char, max_len = make_loaders(
            name_dataset=config.name_dataset,
            split=config.split,
            batch_size=config.batch_size,
            img_size=config.img_size
        )

        # 2. Crear model
        print("\n--- Creant model ---")
        model = MoleculeModel(
            encoder=config.encoder, 
            vocab_size=vocab_size,
            max_len=max_len,
            idx2char=idx2char,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim
        ).to(device) #Puja model a GPU/CPU

        # Mostra informació de cada capa del model (parametres entrenables) i total
        # print(f"Descripció de capes del model:\n{model}") 
        summary(model)

        # Només entrena els parametres amb requires_grad=True
        params = [param for param in model.parameters() if param.requires_grad]
        # print(f"Paràmetres entrenables: {params:,}")

        # 3. Loss i optimizer
        # ignore_index=0 → no penalitza el padding <PAD>
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(
            params,
            lr=config.learning_rate
        )

        # 4. Entrenar
        print("\n--- Iniciant entrenament ---")
        train(model, train_loader, val_loader, optimizer, criterion, params,
              num_epochs=config.epochs, device=device, idx2char=idx2char)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        # Model options
        encoder="Resnet50",
        decoder="LSTM",
        embed_dim=256,
        hidden_dim=512,
        img_size=224,

        # Train parameters
        epochs=20,
        batch_size=16,
        learning_rate=1e-3,

        # Data parameters
        name_dataset="docling-project/USPTO-30K",
        split="clean",

        # Notes
        notes="Capes Resnet50 (-1) congelades"
    )

    model = model_pipeline(config)
