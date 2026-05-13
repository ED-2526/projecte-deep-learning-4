import wandb
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def train_epoch(model, loader, optimizer, criterion, params, device):
    model.train()
    total_loss = 0

    for images, captions, true_len in tqdm(loader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)
        # true_len = true_len.to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs: (batch, seq_len, vocab_size)
        # captions: (batch, seq_len)
        outputs = model(images, captions)
        batch_size, seq_len, vocab_size = outputs.shape

        # Calculem la loss:
        # outputs els aplanem a (batch*seq_len, vocab_size)
        # captions objectiu: saltem el <SOS> inicial → captions[:, 1:]
        target = captions[:, 1:]  # mateixa longitud que outputs
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            target.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping (evita exploding gradients a la RNN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item() #.item() → converteix tensor a número Python

    return total_loss / len(loader) #loss mitjana de l’epoch


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions, true_len in tqdm(loader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)
            # true_len = true_len.to(device)

            outputs = model(images, captions)
            batch_size, seq_len, vocab_size = outputs.shape

            target = captions[:, 1:]  # mateixa longitud que outputs
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                target.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, criterion, params, num_epochs, device, idx2char):
    wandb.watch(model, criterion, log="all", log_freq=50)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, params, device)
        val_loss = val_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
        })

        # Cada 5 epochs, genera un exemple per veure com va
        if (epoch + 1) % 5 == 0:
            print(f"\n  → Avaluant mètriques de molècules...")
            evaluate_molecules(
                model, val_loader, model.idx2char, device,
                num_samples=64,
                epoch=epoch + 1,
            )

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
 
 
def evaluate_molecules(model, loader, idx2char, device, num_samples, epoch=None):
    """
    Genera prediccions sobre `num_samples` molècules del loader i calcula:
      - Tanimoto mitjà
      - % de SMILES vàlids
      - Exact match accuracy
      - Top-k molècules on falla més (menor Tanimoto)
 
    Retorna un dict de mètriques.
    """
    model.eval()

    results = [] 
    samples_done = 0
    with torch.no_grad():
        for images, captions, true_len in loader:
            if samples_done >= num_samples:
                break
 
            images = images.to(device)
            captions = captions.to(device)
 
            batch_size = images.size(0)
            remaining = num_samples - samples_done
            images = images[:remaining]
            captions = captions[:remaining]
            true_len = true_len[:remaining]
 
            for i in range(images.size(0)):
                pred_smiles = model.generate(images[i], model.idx2char, device=device)
 
                # Reconstruir el SMILES real des dels tokens
                tokens = captions[i].cpu().tolist()
                true_smiles = ''.join(
                    model.idx2char.get(t, '')
                    for t in tokens
                    if model.idx2char.get(t, '') not in ['<PAD>', '<SOS>', '<EOS>']
                )
 
                tanimoto, valid = compute_fingerprint_tanimoto(pred_smiles, true_smiles)
                exact = (pred_smiles == true_smiles)
 
                results.append({
                    'true': true_smiles,
                    'pred': pred_smiles,
                    'tanimoto': tanimoto,
                    'valid': valid,
                    'exact': exact,
                })
 
            samples_done += batch_size
 
    # ---- Agregar mètriques ----
    tanimotos = [r['tanimoto'] for r in results]
    valids    = [r['valid']    for r in results]
    exacts    = [r['exact']    for r in results]
 
    metrics = {
        'mol/tanimoto_mean':  float(np.mean(tanimotos)),
        'mol/tanimoto_median': float(np.median(tanimotos)),
        'mol/valid_pct':      float(np.mean(valids)) * 100,
        'mol/exact_match_pct': float(np.mean(exacts)) * 100,
    }
 
    # ---- Pitjors molècules (menor Tanimoto) ----
    worst = sorted(results, key=lambda x: x['tanimoto'])[:10]
 
    worst_table = wandb.Table(
        columns=["epoch", "true_smiles", "pred_smiles", "tanimoto", "valid_pred", "exact"]
    )
    for r in worst:
        worst_table.add_data(
            epoch if epoch is not None else -1,
            r['true'],
            r['pred'],
            round(r['tanimoto'], 4),
            r['valid'],
            r['exact'],
        )
 
    wandb.log({**metrics, "mol/worst_molecules": worst_table})
 
    # ---- Print resum ----
    print(f"\n  [Mètriques molècules | {len(results)} mostres]")
    print(f"  Tanimoto mitjà:   {metrics['mol/tanimoto_mean']:.4f}")
    print(f"  Tanimoto mediana: {metrics['mol/tanimoto_median']:.4f}")
    print(f"  SMILES vàlids:    {metrics['mol/valid_pct']:.1f}%")
    print(f"  Exact match:      {metrics['mol/exact_match_pct']:.1f}%")
    print(f"\n  Top-3 pitjors prediccions:")
    for r in worst[:3]:
        print(f"    TRUE: {r['true']}")
        print(f"    PRED: {r['pred']}  (Tanimoto={r['tanimoto']:.3f}, Vàlid={r['valid']})")
        print()
 
    return metrics
