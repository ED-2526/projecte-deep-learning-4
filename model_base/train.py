import wandb
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def train_epoch(model, loader, optimizer, criterion, device, tf_ratio=1.0):
    model.train()
    total_loss = 0
    total_acc = 0

    for images, captions, true_len in tqdm(loader, desc=f"Training (tf={tf_ratio:.2f})"):
        images = images.to(device)
        captions = captions.to(device)
        true_len = true_len.to(device)

        optimizer.zero_grad()

        batch_size = images.size(0)
        seq_len = captions.size(1) - 1

        # Inicialitza amb la imatge
        features = model.encoder(images)
        h, c = model.decoder.init_state(features)

        outputs = []
        input_token = captions[:, 0:1]  # <SOS>

        for t in range(seq_len):
            out, h, c = model.predict(input_token, h, c, features=features)
            outputs.append(out)

            # Teacher forcing: usa token real o predit?
            if torch.rand(1).item() < tf_ratio:
                input_token = captions[:, t+1:t+2]  # token real
            else:
                input_token = torch.argmax(out, dim=2)  # token predit

        output = torch.cat(outputs, dim=1)
        batch_size_out, seq_len_out, vocab_size = output.shape
        target = captions[:, 1:seq_len_out+1]

        equiv = (torch.argmax(output, dim=2) == target)
        acc = 0
        for ex_len, comparacio in zip(true_len, equiv):
            acc += (torch.sum(comparacio[:ex_len+1]) / ex_len)

        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.params_train, max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item() / batch_size

    return total_loss / len(loader), total_acc / len(loader)


def val_epoch(epoch, model, loader, criterion, device, beam_size=1):
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

    # Cada epoch, es miren les mètriques de l'últim batch del epoch
        print(f"\n  → Fent inferència de molècules...")
        molecule_inference(model, images, captions, epoch, device=device, beam_size=beam_size)

    return total_loss / len(loader), total_acc/len(loader)


def train(model, train_loader, val_loader, optimizer, criterion, config, device):
    wandb.watch(model, criterion, log="all", log_freq=50)
    best_val_loss = float('inf')

    tf_ratio = 1.0

    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")

        if config.teacher_forcing:
            # decaïment en escales:
            tf_ratio = 1- np.floor(epoch/14)*0.1
        else:
            # Sense Teacher Forcing: el model sempre usa el seu propi token
            tf_ratio = 0.0  

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            tf_ratio=tf_ratio
        )
        val_loss, val_acc = val_epoch(epoch, model, val_loader, criterion, device,
                                      beam_size=config.beam_size)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
        print(f"Teacher Forcing: {tf_ratio:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  → Millor model guardat!")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "teacher_forcing_ratio": tf_ratio,
        }, step=epoch+1)
        
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
 
 
def molecule_inference(model, images, captions, epoch, device='cuda', beam_size=1):
    """
    Genera prediccions sobre `num_samples` molècules del loader i calcula:
      - Tanimoto mitjà
      - % de SMILES vàlids
      - Exact match accuracy
      - Top-k molècules on falla més (menor Tanimoto)
 
    Retorna un dict de mètriques.
    """
    
    # Reconstruir el SMILES predits
    if beam_size > 1:
        pred_smiles = [model.generate_beam(image, device=device, beam_size=beam_size) for image in images]
    else:
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

    wandb.log({**metrics, "mol/molecules": table}, step=epoch+1)
    # print(f"\tTanimoto: {tanimoto}")
    # print(f"\tValid: {valid}")
    # print(f"\tExact match: {exact}")
    # print(f"\tPred: {pred_smiles}")
    # print(f"\tTrue: {true_smiles}")
