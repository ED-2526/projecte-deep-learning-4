import argparse
import numpy as np
import random
import time

import wandb 
import torch 
import torch.nn as nn 

from train import train
from utils import make

# Reproduïbilitat
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usant device: {device}")
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()

def model_pipeline(cfg):
    with wandb.init(project="molecule-recognition", config=cfg, name=cfg["name"], group=cfg["group"]):
        config = wandb.config

        # 1. Preparar objectes
        print("\n--- Carregant dades ---")
        model, train_loader, val_loader, criterion, optimizer = make(config, device=device)

        # 2 Fer l'entrenament
        print("\n--- Iniciant entrenament ---")
        train(model, train_loader, val_loader, optimizer, criterion, config, device=device)
        
    return model

if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser()

    # Opcions del model
    parser.add_argument('--encoder', type=str, default='resnet50', 
                        choices=['conv', 'resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('--decoder', type=str, default='lstm', choices=['lstm'])

    group = parser.add_argument_group('embedding_options')
    parser.add_argument('--caption_embed_dim', type=int, default=256)

    group = parser.add_argument_group('cnn_options')
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--image_embed_dim', type=int, default=256)
    parser.add_argument('--unfreeze', default=0, choices=[0, 1, 2, 3, 4]) #0: Cap capa descongelada
    
    group = parser.add_argument_group('lstm_options')
    group.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--decoder_dropout', type=float, default=0.3)
    group.add_argument('--num_layers', type=int, default=1)

    # Parametres entrenament
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--criterion', type=str, default='cross-entropy', 
                        choices=['cross-entropy', 'custom-cross-entropy'])
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--teacher_forcing', action='store_true', default=False)
    # parser.add_argument('--teacher_forcing_schedule', type=str, default='lineal', 
    #                     choices=['lineal', 'sigmoid'])
    
    # Si passes --teacher_forcing, s'activa el decaïment
    # Si no el passes, tf=0.0 sempre
    parser.add_argument('--beam_size', type=int, default=1)
    # beam_size=1 és equivalent a greedy (per defecte)

    # Dataset
    parser.add_argument('--dataset', type=str, default='principal', 
                        choices=['principal', 'all'])
    parser.add_argument('--split', type=str, default='clean',
                        choices=['clean', 'abbreviated', 'large'])
    parser.add_argument('--train_percentage', type=float, default=0.8)
    parser.add_argument('--image_channels', type=int, default=3)

    parser.add_argument('--smiles_filter', action='store_true', default=False)
    parser.add_argument('--min_smiles_len', default=40)
    parser.add_argument('--max_smiles_len', default=60)
    
    # Notes wandb
    parser.add_argument('--name', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--description', default='')

    args = parser.parse_args()
    start = time.time()
    model = model_pipeline(vars(args))
    end = time.time()
    print(f"\nTemps d'execució: {(end-start)/60} minuts")
