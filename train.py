import torch
import wandb
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import editdistance

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, captions, _ in tqdm(loader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        outputs = model(images, captions)

        batch_size, seq_len, vocab_size = outputs.shape
        target = captions[:, 1:seq_len+1]
        loss = criterion(outputs.reshape(-1, vocab_size), target.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)

            batch_size, seq_len, vocab_size = outputs.shape
            target = captions[:, 1:seq_len+1]
            loss = criterion(outputs.reshape(-1, vocab_size), target.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, criterion,
          num_epochs, device, idx2char, scheduler=None):

    wandb.watch(model, criterion, log="all", log_freq=50)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  → Millor model guardat!")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if (epoch + 1) % 5 == 0:
            sample_img = next(iter(val_loader))[0][0]
            predicted = model.generate(sample_img, idx2char, device=device)
            print(f"\n  Exemple generat (primers 200 cars):")
            print(f"  {predicted[:200]}")

            evaluate_molecules(model, val_loader, idx2char,
                               device, num_samples=64, epoch=epoch+1)


def compute_fingerprint_tanimoto(smiles_pred, smiles_true):
    mol_true = Chem.MolFromSmiles(smiles_true)
    mol_pred = Chem.MolFromSmiles(smiles_pred)
    valid_pred = mol_pred is not None
    if mol_true is None or mol_pred is None:
        return 0.0, valid_pred
    fp_true = _morgan_gen.GetFingerprint(mol_true)
    fp_pred = _morgan_gen.GetFingerprint(mol_pred)
    return DataStructs.TanimotoSimilarity(fp_true, fp_pred), valid_pred


def evaluate_molecules(model, loader, idx2char, device, num_samples, epoch=None):
    model.eval()
    results = []
    samples_done = 0

    with torch.no_grad():
        for images, captions, _ in loader:
            if samples_done >= num_samples:
                break
            images = images.to(device)
            captions = captions.to(device)
            remaining = num_samples - samples_done
            images = images[:remaining]
            captions = captions[:remaining]

            for i in range(images.size(0)):
                pred_smiles = model.generate(images[i], idx2char, device=device)
                tokens = captions[i].cpu().tolist()
                true_smiles = ''.join(
                    idx2char.get(t, '')
                    for t in tokens
                    if idx2char.get(t, '') not in ['<PAD>', '<SOS>', '<EOS>']
                )
                tanimoto, valid = compute_fingerprint_tanimoto(pred_smiles, true_smiles)
                exact = (pred_smiles == true_smiles)

                # ← Edit distance: quants caràcters cal canviar
                edit_dist = editdistance.eval(pred_smiles, true_smiles)
                edit_dist_norm = edit_dist / max(len(true_smiles), 1)

                results.append({
                    'true': true_smiles,
                    'pred': pred_smiles,
                    'tanimoto': tanimoto,
                    'valid': valid,
                    'exact': exact,
                    'edit_dist': edit_dist,
                    'edit_dist_norm': edit_dist_norm,
                })
            samples_done += images.size(0)

    tanimotos  = [r['tanimoto']       for r in results]
    valids     = [r['valid']          for r in results]
    exacts     = [r['exact']          for r in results]
    edit_dists = [r['edit_dist_norm'] for r in results]  # ← nou

    metrics = {
        'mol/tanimoto_mean':   float(np.mean(tanimotos)),
        'mol/tanimoto_median': float(np.median(tanimotos)),
        'mol/valid_pct':       float(np.mean(valids)) * 100,
        'mol/exact_match_pct': float(np.mean(exacts)) * 100,
        'mol/edit_dist_norm':  float(np.mean(edit_dists)),  # ← nou
    }

    worst = sorted(results, key=lambda x: x['tanimoto'])[:10]
    worst_table = wandb.Table(
        columns=["epoch", "true_smiles", "pred_smiles",
                 "tanimoto", "valid_pred", "exact", "edit_dist"]
    )
    for r in worst:
        worst_table.add_data(
            epoch if epoch is not None else -1,
            r['true'], r['pred'],
            round(r['tanimoto'], 4), r['valid'], r['exact'],
            r['edit_dist']  # ← nou
        )
    wandb.log({**metrics, "mol/worst_molecules": worst_table})

    print(f"\n  [Mètriques molècules | {len(results)} mostres]")
    print(f"  Tanimoto mitjà:     {metrics['mol/tanimoto_mean']:.4f}")
    print(f"  Tanimoto mediana:   {metrics['mol/tanimoto_median']:.4f}")
    print(f"  SMILES vàlids:      {metrics['mol/valid_pct']:.1f}%")
    print(f"  Exact match:        {metrics['mol/exact_match_pct']:.1f}%")
    print(f"  Edit Distance norm: {metrics['mol/edit_dist_norm']:.4f}")  # ← nou
    print(f"\n  Top-3 pitjors prediccions:")
    for r in worst[:3]:
        print(f"    TRUE: {r['true']}")
        print(f"    PRED: {r['pred']}  "
              f"(Tanimoto={r['tanimoto']:.3f}, "
              f"Edit={r['edit_dist']}, "
              f"Vàlid={r['valid']})")
        print()

    analyze_errors_by_atom(results)
    return metrics


def analyze_errors_by_atom(results):
    """Agrupa els errors per tipus d'àtom predominant"""
    from rdkit import Chem
    
    groups = {'amb_O': [], 'amb_N': [], 'amb_anell': [], 'altres': []}
    
    for r in results:
        mol = Chem.MolFromSmiles(r['true'])
        if mol is None:
            continue
        
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        n_O = atoms.count('O')
        n_N = atoms.count('N')
        has_ring = mol.GetRingInfo().NumRings() > 0
        
        if n_O > n_N and n_O > 2:
            groups['amb_O'].append(r['tanimoto'])
        elif n_N > n_O and n_N > 2:
            groups['amb_N'].append(r['tanimoto'])
        elif has_ring:
            groups['amb_anell'].append(r['tanimoto'])
        else:
            groups['altres'].append(r['tanimoto'])
    
    print("\n  [Anàlisi per tipus de molècula]")
    for grup, tans in groups.items():
        if tans:
            print(f"  {grup}: {len(tans)} mostres | Tanimoto mitjà: {np.mean(tans):.4f}")
    
    return groups