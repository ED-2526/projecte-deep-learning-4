import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')  

class MoleculeDataset(Dataset):
    def __init__(self, split='clean', max_len=120, img_size=224):
        print(f"Carregant split '{split}'...")
        raw_data = load_dataset("docling-project/USPTO-30K")[split]
        self.max_len = max_len

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Convertir MolFiles a SMILES usando RDKit
        print("Convertint MolFiles a SMILES...")
        self.data = []
        skipped = 0
        for item in raw_data:
            mol = Chem.MolFromMolBlock(item['mol'], sanitize=True)
            if mol is None:
                skipped += 1
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)  # SMILES canònic
            if smiles and len(smiles) <= max_len:
                self.data.append({'image': item['image'], 'smiles': smiles})

        print(f"Mostres vàlides: {len(self.data)} | Descartades: {skipped}")

        # Construir vocabulari sobre SMILES (molt més petit que MolFile)
        print("Construint vocabulari...")
        all_text = [item['smiles'] for item in self.data]
        chars = sorted(set(''.join(all_text)))

        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, c in enumerate(chars):
            self.char2idx[c] = i + 3

        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"Vocabulari: {self.vocab_size} caràcters únics")
        print(f"Exemples finals: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item['image'].convert('RGB'))

        smiles = item['smiles']
        tokens = (
            [self.char2idx['<SOS>']] +
            [self.char2idx.get(c, 0) for c in smiles] +
            [self.char2idx['<EOS>']]
        )

        pad_len = self.max_len + 2 - len(tokens)
        tokens = tokens + [self.char2idx['<PAD>']] * pad_len

        return image, torch.tensor(tokens, dtype=torch.long)


def make_loaders(batch_size=16, max_len=120, img_size=224):
    train_dataset = MoleculeDataset(split='clean', max_len=max_len, img_size=img_size)

    total = len(train_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size

    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {train_size} mostres | Val: {val_size} mostres")
    return train_loader, val_loader, train_dataset.vocab_size, train_dataset.idx2char