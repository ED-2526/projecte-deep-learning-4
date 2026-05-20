# Molecule Recognition
L'objectiu d'aquest projecte és aconseguir un model que, donat la imatge d'una molècula, doni la seva fòrmulació correcta. 

## Model 
En el primer model, s'ha fet servir una CNN preentrenada per extreure la informació important de les imatges (ResNet18, ResNet50 i ResNet101) i una RNN de tipus LSTM per aconseguir la SMILES corresponent. La funció de cost és la CrossEntropyLoss, calculada per cada caràcter del batch. 


## Datasets
- Molècules referència (principal): https://huggingface.co/datasets/docling-project/USPTO-30K
- Molècules sintètiques (opcional): https://huggingface.co/datasets/docling-project/MolGrapher-Synthetic-300K

## Environment
Crear environment:
```
conda env create --file environment.yml
``` 

Activar environment: 
```
conda activate deeplearning
```

Per executar:
```
python main.py
```

## Contributors
- Mar Massanas
- David Liu
- Daniela Lou Pardo

Xarxes Neuronals i Aprenentatge Profund
Grau d'Énginyeria de Dades, 
UAB, 2026
