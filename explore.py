from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

print("Carregant dataset...")
dataset = load_dataset("docling-project/USPTO-30K")

print("Splits disponibles:", dataset)
print("Clean:", len(dataset['clean']))
print("Abbreviated:", len(dataset['abbreviated']))
print("Large:", len(dataset['large']))

exemple = dataset['clean'][0]
print("\nClaus de cada exemple:", exemple.keys())
print("\nNom del fitxer:", exemple['filename'])
print("\nText MolFile (primers 300 caràcters):")
print(exemple['mol'][:300])

exemple['image'].save('exemple_molecula.png')
print("\nImatge guardada com exemple_molecula.png")

imatges_mides = [dataset['clean'][i]['image'].size for i in range(10)]
print("\nMides de les primeres 10 imatges:", imatges_mides)

longituds = [len(dataset['clean'][i]['mol']) for i in range(100)]
print(f"\nLongitud text MolFile - Mínim: {min(longituds)}, Màxim: {max(longituds)}, Mitjana: {sum(longituds)//len(longituds)}")
