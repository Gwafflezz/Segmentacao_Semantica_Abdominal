[[]()]()---
tags: 
Título: 
Local: 
Data: 
Subprojeto: 
Participantes:
---
Projeto:
Referências:  


---
Recebe um caminho de diretório,uma lista vazia e  uma tupla com
dimensões de imagem. Lê as imagens png ou jpg do diretório, ordena pelo nome e
 armazena a imagem no array img_data e seu nome na lista img_names.
 
---

```python

"""esta função segmenta o nome do arquivo para o img_loader ordenar o dataset
na ordem do diretório e ter correspondência entre a lista de imagens e máscaras"""

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

""" img_loader recebe um caminho de diretório,uma lista vazia e  uma tupla com
dimensões de imagem. Lê as imagens png ou jpg do diretório, ordena pelo nome e
 armazena a imagem no array img_data e seu nome na lista img_names."""
def img_loader(path, img_data, size=None, rgb=True):
  #lista para o nome dos arquivos
  img_names = []

  for diretorio_path in sorted(glob.glob(path)):
    for img_path in sorted(glob.glob(os.path.join(diretorio_path, "*.[pj]*[np]*[g]*")), key=natural_sort_key): #percorre o diretório na ordem natural dos títulos de arquivo
      img = cv2.imread(img_path,
                       cv2.IMREAD_COLOR if rgb
                       else cv2.IMREAD_GRAYSCALE) #img tem 3 canais na 3 dimensao se RGB, e 1 canal se preto/branco

      if size is not None:
        img = cv2.resize(img, size) #redimensiona conforme o parâmetro

      img_data.append(img.astype(np.uint8)) #add a imagem na lista do parametro
      img_names.append(os.path.basename(img_path)) #add o nome do arquivo na lista de nomes

  #return img_data, img_names
  return np.array(img_data), img_names
```

