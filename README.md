# Molecule Recognition
L'objectiu d'aquest projecte és aconseguir un model que, donat la imatge d'una molècula, doni la seva fòrmulació correcta. 

## Model 
En un principi s'ha fet servir una CNN preentrenada per extreure la informació important de les imatges (ResNet18, ResNet50 i ResNet101) i una RNN per aconseguir el text. 

(per posar: foto model)

## Datasets
- Molècules referència (principal): https://huggingface.co/datasets/docling-project/USPTO-30K
- Molècules sintètiques (secundari): https://huggingface.co/datasets/docling-project/MolGrapher-Synthetic-300K

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
