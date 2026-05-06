import torch
import torch.nn as nn
import wandb
import random
import numpy as np

from models.models import MoleculeModel
from utils.utils import make_loaders
from train import train

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
            batch_size=config.batch_size,
            img_size=config.img_size
        )

        # 2. Crear model
        print("\n--- Creant model ---")
        model = MoleculeModel(
            max_len=max_len,
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim, 
        ).to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paràmetres entrenables: {params:,}")

        # 3. Loss i optimizer
        # ignore_index=0 → no penalitza el padding <PAD>
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        # 4. Entrenar
        print("\n--- Iniciant entrenament ---")
        train(model, train_loader, val_loader, optimizer, criterion,
              num_epochs=config.epochs,
              device=device,
              idx2char=idx2char)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=20,
        batch_size=16,
        learning_rate=1e-3,
        embed_dim=256,
        hidden_dim=512,
        img_size=224,
        dataset="USPTO-30K-clean",
        architecture="ResNet18+LSTM"
    )

    model = model_pipeline(config)
