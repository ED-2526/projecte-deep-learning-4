import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import wandb

# ============================================================
# DATASET: carrega USPTO-30K i converteix imatges i text
# ============================================================

class MoleculeDataset(Dataset):
    def __init__(self, split='clean', max_len=500, img_size=224):
        """
        split: 'clean', 'abbreviated' o 'large'
        max_len: longitud màxima del text MolFile (trunca els més llargs)
        img_size: mida de la imatge quadrada
        """
        print(f"Carregant split '{split}'...")
        self.data = load_dataset("docling-project/USPTO-30K")[split]
        self.max_len = max_len
        
        # Transformació de la imatge:
        # 1. Redimensiona a img_size x img_size (totes han de ser iguals)
        # 2. Converteix a escala de grisos (les molècules són en blanc i negre)
        # 3. Converteix a tensor PyTorch
        # 4. Normalitza els píxels (millora l'entrenament)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Construir vocabulari de caràcters
        # El model no treballa amb text directament, sinó amb números
        # Cada caràcter únic del dataset rep un número (índex)
        print("Construint vocabulari...")
        all_text = [item['mol'] for item in self.data]
        chars = sorted(set(''.join(all_text)))
        
        # Tokens especials:
        # <PAD> = farciment per igualar longituds
        # <SOS> = Start Of Sequence (inici de seqüència)
        # <EOS> = End Of Sequence (fi de seqüència)
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, c in enumerate(chars):
            self.char2idx[c] = i + 3
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"Vocabulari: {self.vocab_size} caràcters únics")
        print(f"Exemples al split: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Processar la imatge
        image = self.transform(item['image'].convert('RGB'))
        
        # Processar el text: convertir caràcters a índexs numèrics
        mol_text = item['mol'][:self.max_len]  # trunca si és massa llarg
        tokens = (
            [self.char2idx['<SOS>']] +
            [self.char2idx.get(c, 0) for c in mol_text] +
            [self.char2idx['<EOS>']]
        )
        
        # Farciment (padding) per igualar longituds dins el batch
        pad_len = self.max_len + 2 - len(tokens)
        tokens = tokens + [self.char2idx['<PAD>']] * pad_len
        
        return image, torch.tensor(tokens, dtype=torch.long)


def make_loaders(batch_size=16, max_len=500, img_size=224):
    """Crea els DataLoaders de train i validació"""
    
    # Usem el split 'clean' (les molècules més senzilles) per començar
    train_dataset = MoleculeDataset(split='clean', 
                                     max_len=max_len, 
                                     img_size=img_size)
    
    # Dividim manualment en train (80%) i validació (20%)
    total = len(train_dataset)
    train_size = int(0.8 * total)  # 8.000 mostres
    val_size = total - train_size   # 2.000 mostres
    
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
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
    return train_loader, val_loader, train_dataset.vocab_size, train_dataset.idx2char
