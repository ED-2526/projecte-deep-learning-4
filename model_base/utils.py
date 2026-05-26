import wandb
from datasets import load_dataset #Hugging Face datasets. descarrega USPTO-30K
from rdkit import Chem #RDKit: processar molècules. convertir MolFiles a SMILES
import torch
from torchinfo import summary
import torch.nn as nn
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader 

from molecule_dataset import MoleculeDataset
from models import MoleculeModel, MoleculeDecoder, MoleculeEncoder


def make_criterion(criterion, label_smoothing, vocab, device):
     """Crea un criterion per calcular la loss. """
     if criterion == 'custom-cross-entropy':
        # Crea vector de pesos per a cada token del vocabulari
        vocab_size = len(vocab)
        weights = torch.ones(vocab_size)

        # PAD no compta
        # weights[vocab['<PAD>']] = 0.0

        # Penalitza molt poc confondre C amb c (tots dos son carboni)
        # Penalitza el doble si confon C amb N, F, O, etc.
        # (no podem fer-ho exacte aquí, però pujem el pes dels heteroàtoms)
        heteroatoms = ['N', 'O', 'F', 'S', 'Cl', 'Br', 'n', 'o', 's', 'P']
        for atom in heteroatoms:
            if atom in vocab:
                weights[vocab[atom]] = 2.0

        weights = weights.to(device)
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
        )
     else: 
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
def make_loaders(dataset, batch_size, train_percentage):
    """Funció per aconseguir els train i val loaders segons un percentatge.

    Args:
        dataset (MoleculeDataset): dataset de les molècules.
        batch_size (int): mida del batch size.
        train_percentage (float): percentatge del train dataset.

    Returns:
        train_loader (torch.loader): loader de train.
        val_loader (torch.loader): loader de validació.
    """

    # Dividim manualment en train (80%) i validació (20%)
    total = len(dataset)
    train_size = int(train_percentage* total)  # 8.000 mostres
    val_size = total - train_size   # 2.000 mostres
    
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True 
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train: {train_size} mostres | Val: {val_size} mostres")
    return train_loader, val_loader

def make(config, device='cuda'):
    """Funció per crear el model, els loaders, el criterion i el optimizer.

    Args:
        config (dict): diccionari de configuració amb totes les opcions.
        device (str, optional): dispositiu on es calcula.

    Returns:
        model (nn.Module): model que ajunta l'encoder i el decoder. 
        train_loader (torch.loader): loader d'entrenament.
        val_loader (torch.loader): loader de validació.
        criterion (nn.Module): criterion per la loss.
        optimizer (torch.optim): per distribuir el gradient.
    """
    
    dataset = MoleculeDataset(config.dataset, config.split, config.image_channels, config.input_dim, config.smiles_filter,
                              min_smiles_len=config.min_smiles_len,
                              max_smiles_len=config.max_smiles_len)
    train_loader, val_loader = make_loaders(dataset, config.batch_size, config.train_percentage)
    
    model = MoleculeModel(config.encoder, config.image_embed_dim, config.image_embed_dim,
                          config.hidden_dim, config.unfreeze4, dataset.vocab_size, dataset.max_len, 
                          dataset.diccionaris(), config.decoder_dropout,
                          num_layers=config.num_layers).to(device)

    summary(model)

    criterion = make_criterion(config.criterion, config.label_smoothing, dataset.char2idx, device)

    # Només entrena els parametres amb requires_grad=True
    optimizer = torch.optim.Adam(model.params_train, lr=config.learning_rate)

    return model, train_loader, val_loader, criterion, optimizer