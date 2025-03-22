---
tags: 
Título: 
Local: 
Data: 
Subprojeto: 
Participantes:
---
---
Mapeia cada valor RGB para um índice de 0 a 8:
Retorna um dicionário de tuplas > inteiros.

---


```python
def generate_mask_dict():
    mask_dict = {
        #valores rgb dos pixels:
        (0, 0, 0): 0,        # Preto = None
        (100, 0, 100): 1,    # violeta = Liver(fígado)
        (255, 255, 255): 2,  # Branco = Bone(Osso)
        (0, 255, 0): 3,      # Verde = Gallbladder(Vesícula biliar)
        (255, 255, 0): 4,    # Amarelo = Kidney(Rins)
        (0, 0, 255): 5,      # Azul = Pancreas
        (255, 0, 0): 6,      # Vermelho = Vessels(Veias)
        (255, 0, 255): 7,    # Rosa = Spleen(Baço)
        (0, 255, 255): 8     # Azul claro = Adrenal(Gland. Adrenal)
    }
    return mask_dict
```
