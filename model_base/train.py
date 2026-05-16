import wandb
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    
    total_loss = 0
    total_acc = 0

    for images, captions, true_len in tqdm(loader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)
        true_len = true_len.to(device)

        optimizer.zero_grad()

        # Forward pass
        # captions: (batch, max_len)
        # output: (batch, max_len, vocab_size)
        output, h, c = model(images, captions[:, :-1])
        batch_size, seq_len, vocab_size = output.shape

        # Calculem la loss:
        # output els aplanem a (batch*seq_len, vocab_size)
        # captions objectiu: saltem el <SOS> inicial → captions[:, 1:]
        target = captions[:, 1:]  # mateixa longitud que output

        equiv = (torch.argmax(output, dim=2) == target)
        acc = 0
        # acc = torch.equal(torch.argmax(output, dim=2), target)
        for ex_len, comparacio in zip(true_len, equiv): 
            acc += (torch.sum(comparacio[:ex_len+1])/ex_len)

        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping (evita exploding gradients a la RNN)
        torch.nn.utils.clip_grad_norm_(model.params_train, max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() #.item() → converteix tensor a número Python
        total_acc += acc.item()/batch_size
        # total_acc += acc

    return total_loss/len(loader), total_acc/len(loader) #loss mitjana de l’epoch


def val_epoch(epoch, model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for images, captions, true_len in tqdm(loader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)
            true_len = true_len.to(device)

            output, h, c = model(images, captions[:, :-1])
            batch_size, seq_len, vocab_size = output.shape

            target = captions[:, 1:]  # mateixa longitud que output

            # acc = torch.equal(torch.argmax(output, dim=2), target)

            equiv = (torch.argmax(output, dim=2) == target)
            acc = 0
            for ex_len, comparacio in zip(true_len, equiv): 
                acc += (torch.sum(comparacio[:ex_len+1])/ex_len)

            loss = criterion(
                output.reshape(-1, vocab_size),
                target.reshape(-1)
            )

            total_loss += loss.item()
            total_acc += acc.item()/batch_size
            # total_acc += acc

    # Cada 5 epochs, es miren les mètriques de l'últim batch del epoch
        if (epoch + 1) % 5 == 0:
            print(f"\n  → Fent inferència de molècules...")
            molecule_inference(model, images, captions, epoch, device=device)

    return total_loss / len(loader), total_acc/len(loader)


def train(model, train_loader, val_loader, optimizer, criterion, config, device):
    wandb.watch(model, criterion, log="all", log_freq=50)

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,  val_acc = val_epoch(epoch, model, val_loader, criterion, device)

        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Guardar el millor model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  → Millor model guardat!")

        # Log a WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

def compute_fingerprint_tanimoto(smiles_pred, smiles_true):
    """
    Calcula la similitud de Tanimoto entre dues molècules via Morgan fingerprints.
    Retorna:
      - tanimoto (float): 0.0 si alguna molècula és invàlida
      - valid_pred (bool): si el SMILES predicat és químicament vàlid
    """
    mol_true = Chem.MolFromSmiles(smiles_true)
    mol_pred = Chem.MolFromSmiles(smiles_pred)
 
    valid_pred = mol_pred is not None
 
    if mol_true is None or mol_pred is None:
        return 0.0, valid_pred
    
    fp_true = _morgan_gen.GetFingerprint(mol_true)
    fp_pred = _morgan_gen.GetFingerprint(mol_pred)
    
    return DataStructs.TanimotoSimilarity(fp_true, fp_pred), valid_pred
 
 
def molecule_inference(model, images, captions, epoch, device='cuda'):
    """
    Genera prediccions sobre `num_samples` molècules del loader i calcula:
      - Tanimoto mitjà
      - % de SMILES vàlids
      - Exact match accuracy
      - Top-k molècules on falla més (menor Tanimoto)
 
    Retorna un dict de mètriques.
    """
    
    # Reconstruir el SMILES predits
    pred_smiles = [model.generate_prediction(image, device=device) for image in images]

    # Reconstruir el SMILES real des dels tokens
    true_smiles = [model.generate_smiles(caption) for caption in captions]
    
    tanimotos = []
    valids = []
    exacts = []    

    for pred_smiles_exemple, true_smiles_exemple in zip(pred_smiles, true_smiles): 
        tanimoto, valid = compute_fingerprint_tanimoto(pred_smiles_exemple, true_smiles_exemple)
        exact = (pred_smiles_exemple == true_smiles_exemple)

        tanimotos.append(tanimoto)
        valids.append(valid)
        exacts.append(exact)
 
    metrics = {
        'epoch': epoch+1,
        'mol/tanimoto_mean':  (np.mean(tanimotos)),
        'mol/valid_pct':      (np.mean(valids)),
        'mol/exact_match_pct': (np.mean(exacts)),
    }
    
    print(metrics)

    table = wandb.Table(
        columns=['epoch', 'true_smiles', 'pred_smiiles', 'tanimoto', 'valid_pred', 'exact']
    )
    
    for i in range(len(images)): 
        table.add_data(epoch+1, true_smiles[i], pred_smiles[i], tanimotos[i], valids[i], exacts[i])
        if i < 3: 
            print(f"\n\tTRUE:\n\t{true_smiles[i]}")
            print(f"\tPRED:\n\t{pred_smiles[i]}")

    wandb.log({**metrics, "mol/molecules": table})
    # print(f"\tTanimoto: {tanimoto}")
    # print(f"\tValid: {valid}")
    # print(f"\tExact match: {exact}")
    # print(f"\tPred: {pred_smiles}")
    # print(f"\tTrue: {true_smiles}")
