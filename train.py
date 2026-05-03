import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors


# ============================================================
# MÉTRICAS QUÍMICAS
# ============================================================

def mol_from_text(text):
    try:
        return Chem.MolFromSmiles(text)  # ← SMILES, no MolBlock
    except:
        return None

def tanimoto_similarity(pred_text, true_text):
    """Calcula Tanimoto similarity entre dos MolFiles. Retorna 0.0 si inválido."""
    mol_pred = mol_from_text(pred_text)
    mol_true = mol_from_text(true_text)
    if mol_pred is None or mol_true is None:
        return 0.0
    try:
        fp_pred = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_pred, radius=2)
        fp_true = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_true, radius=2)
        return DataStructs.TanimotoSimilarity(fp_pred, fp_true)
    except:
        return 0.0

def is_valid_mol(text):
    """Retorna True si el texto es un MolFile parseable."""
    return mol_from_text(text) is not None

def decode_tokens(token_ids, idx2char):
    """Convierte lista de índices a string, parando en <EOS>."""
    chars = []
    for idx in token_ids:
        char = idx2char.get(idx, '')
        if char == '<EOS>':
            break
        if char not in ('<PAD>', '<SOS>'):
            chars.append(char)
    return ''.join(chars)

def compute_chemistry_metrics(model, loader, idx2char, device, num_samples=64):
    model.eval()
    valid_count = 0
    tanimoto_scores = []
    tanimoto_valid_scores = []
    evaluated = 0

    with torch.no_grad():
        for images, captions in loader:
            if evaluated >= num_samples:
                break

            batch_size = min(images.size(0), num_samples - evaluated)
            images = images[:batch_size].to(device)
            captions = captions[:batch_size]

            for i in range(batch_size):
                pred_text = model.generate(images[i], idx2char, device=device)

                # Reconstruir SMILES real saltando <SOS> (índex 1)
                # i parant a <EOS> (índex 2) o <PAD> (índex 0)
                true_tokens = captions[i].tolist()
                true_chars = []
                for tok in true_tokens:
                    if tok in (0, 2):  # <PAD> o <EOS>
                        break
                    if tok == 1:       # <SOS> — skip
                        continue
                    c = idx2char.get(tok, '')
                    if c not in ('<PAD>', '<SOS>', '<EOS>'):
                        true_chars.append(c)
                true_text = ''.join(true_chars)

                valid = is_valid_mol(pred_text)
                if valid:
                    valid_count += 1

                score = tanimoto_similarity(pred_text, true_text)
                tanimoto_scores.append(score)
                if valid:
                    tanimoto_valid_scores.append(score)

                # Debug primers exemples
                if evaluated < 3:
                    print(f"\n[{evaluated}] TRUE:  {true_text[:80]}")
                    print(f"[{evaluated}] PRED:  {pred_text[:80]}")
                    print(f"[{evaluated}] valid={valid} tanimoto={score:.4f}")

                evaluated += 1

    pct_valid = valid_count / evaluated * 100 if evaluated > 0 else 0
    tanimoto_mean = sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0.0
    tanimoto_valid_mean = (
        sum(tanimoto_valid_scores) / len(tanimoto_valid_scores)
        if tanimoto_valid_scores else 0.0
    )
    return pct_valid, tanimoto_mean, tanimoto_valid_mean


# ============================================================
# TRAIN / VAL EPOCH  
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, captions in tqdm(loader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        outputs = model(images, captions)          # (batch, seq_len-1, vocab_size)
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)), # (batch*(seq_len-1), vocab_size)
            captions[:, 1:].reshape(-1)            # (batch*(seq_len-1),) — skip <SOS>
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                captions[:, 1:].reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          num_epochs, device, idx2char):

    wandb.watch(model, criterion, log="all", log_freq=50)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  → Millor model guardat!")

        log_dict = {
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }

        # Cada 5 epochs: métricas químicas completas
        if (epoch + 1) % 5 == 0:
            print("\n  Calculant mètriques químiques (64 mostres)...")
            pct_valid, tanimoto_mean, tanimoto_valid = compute_chemistry_metrics(
                model, val_loader, idx2char, device, num_samples=64
            )

            print(f"  % MolFiles vàlids:          {pct_valid:.1f}%")
            print(f"  Tanimoto mitjà (tots):      {tanimoto_mean:.4f}")
            print(f"  Tanimoto mitjà (vàlids):    {tanimoto_valid:.4f}")
            print(f"  [TRAIN] % vàlids: {pct_valid:.1f}% | Tanimoto: {tanimoto_mean:.4f}")
            log_dict.update({
                "pct_valid_mols":   pct_valid,
                "tanimoto_mean":    tanimoto_mean,
                "tanimoto_valid":   tanimoto_valid,
            })

            # Ejemplo generado
            sample_img = next(iter(val_loader))[0][0]
            predicted = model.generate(sample_img, idx2char, device=device)
            print(f"\n  Exemple generat (primers 200 chars):")
            print(f"  {predicted[:200]}")
            
        scheduler.step(val_loss)

        wandb.log(log_dict)