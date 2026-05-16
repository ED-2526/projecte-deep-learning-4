from datasets import load_dataset 
from rdkit import Chem 
import torch
from torchvision import transforms 
from torch.utils.data import Dataset 

class Padding: 
    """Classe emprada per MoleculeDataset per calcular i fer el màxim padding de les imatges. 
    """
    def __init__(self): 
        self.height = 0
        self.width = 0

    def __call__(self, image):
        """Funció fer padding a la imatge proporcionada segons la mida màxima.

        Args:
            image (PIL or tensor): imatge per fer padding

        Returns:
            PIL or tensor: Imatge original amb white padding
        """
        h, w = image.size(1), image.size(2)
        hp = int((self.height - h) / 2)
        wp = int((self.width - w) / 2)

        # Afegeix 1 més de padding en el cas que sigui parell
        if hp % 2: hp+=1
        if wp % 2: wp+=1

        padding = (wp, hp)
        t = transforms.Pad(padding, 1)
        return t(image)
        
    def comparar(self, height, width): 
        """Funció per comparar l'altura i mida màxima actual.

        Args:
            height (int): altura.
            width (int): amplada.
        """
        self.height = max(self.height, height)
        self.width = max(self.width, width)

    def max_dimension(self): 
        """Funció per retornar la mida màxima actual.

        Returns:
            (int, int): altura i amplada.
        """
        # Assegurar que la mida final sigui impar
        if not self.height % 2: self.height +=1
        if not self.width % 2: self.width +=1
        
        return (self.height, self.width)

class MoleculeDataset(Dataset):
    """Classer per guardar les dades de les molècules del dataset desitjat.
    """
    def __init__(self, dataset, split, image_channels, input_dim):
        """Càrrega i prepara les dades del dataset.

        Args:
            dataset (str): nom del dataset.
            split (str): split del dataset.
            image_channels (int): número de canals desitjats de la imatge.
            input__dim (int): mida desitjada de la imatge.

        Raises:
            ValueError: si el nom del dataset és incorrecte.
        """

        print(f"Carregant '{dataset}--{split}'...")
    
        max_square = Padding()

        if dataset=="principal": 
            # Aquest dataset té molblocks.
            # Els splits oficials són: 'clean', 'abbreviated', 'large'
            name_dataset = "docling-project/USPTO-30K"
            raw_data = load_dataset(name_dataset)[split]

            # Convertir MolFiles a SMILES amb RDKit
            print("\nConvertint MolFiles a SMILES...")
            self.data = []
            skipped = 0

            for item in raw_data:
                # Comprova si les molècules són vàlides
                mol = Chem.MolFromMolBlock(item['mol'])
                if mol is None:
                    skipped += 1
                    continue
                smiles = Chem.MolToSmiles(mol)  # SMILES canònic

                # Màxima shape
                w, h = item['image'].size
                max_square.comparar(h, w)

                # Afegeix nova molècula
                self.data.append({'image': item['image'], 'smiles': smiles})

            print(f"\tMostres vàlides: {len(self.data)} | Descartades: {skipped}")
        
        elif dataset == "secundari":
            # Aquest dataset ja inclou el camp 'smiles' directament.
            # Els splits oficials són: 'train', 'validation', 'test'
            name_dataset = "docling-project/MolGrapher-Synthetic-300K"
            raw_data = load_dataset(name_dataset)[split]

            print("\nLlegint SMILES directament del dataset...")
            self.data = []
            skipped = 0

            for item in raw_data:
                smiles = item['smiles']

                # Màxima shape
                w, h = item['image'].size
                max_square.comparar(h, w)

                # Afegeix nova molècula
                self.data.append({'image': item['image'], 'smiles': smiles})

            print(f"\tMostres vàlides: {len(self.data)} | Descartades: {skipped}")
        else:
            raise ValueError(f"Dataset no suportat: {dataset}")
        
        # Construir vocabulari de caràcters
        # El model no treballa amb text directament, sinó amb números
        # Cada caràcter únic del dataset rep un número (índex)
        print("\nConstruint vocabulari...")
        self.max_len = 0
        all_text = []
        
        for item in self.data:
            self.max_len = max(self.max_len, len(item["smiles"]))
            all_text.append(item['smiles'])

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
        print(f"Número d'exemples a {name_dataset}--{split}: {len(self.data)}")
        print(f"Mida màxima de SMILES a {name_dataset}--{split}: {self.max_len}")
        print(f"Dimensió màxima (hxw) de Imatge a {name_dataset}--{split}: {max_square.max_dimension()}")

        # Transformacions de les imatges:
        # 1. Converteix a escala de grisos (blanc i negre) amb 3 canals o 1
        # 2. Converteix a tensor PyTorch
        # 3. Padding fins màxim dimensions
        # 4. Elimina padding excessiu amb mides parells
        # 5. Normalitza els píxels (millora l'entrenament)
        self.transformacions = transforms.Compose([
            transforms.Resize((input_dim, input_dim)),
            transforms.Grayscale(num_output_channels=image_channels),
            transforms.ToTensor(),
            # self.max_square, 
            # transforms.CenterCrop((224, 224))
            # transforms.CenterCrop(max_square.max_dimension()), 
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data) #quantes mostres/àtoms hi ha

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Processar la imatge
        image = self.transformacions(item['image']) 
        
        # Processar el text: convertir caràcters a índexs numèrics
        mol_text = item['smiles'] 
        tokens = (
            [self.char2idx['<SOS>']] +
            [self.char2idx.get(c, 0) for c in mol_text] +
            [self.char2idx['<EOS>']]
        )
        
        # Farciment (padding) per igualar longituds dins el batch
        pad_len = self.max_len + 2 - len(tokens)
        tokens = tokens + [self.char2idx['<PAD>']] * pad_len
        
        return image, torch.tensor(tokens, dtype=torch.long), len(item["smiles"])
    
    def diccionaris(self): 
        return self.char2idx, self.idx2char