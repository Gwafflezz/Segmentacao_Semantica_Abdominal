
import tensorflow as tf
import keras
import os
import re
import glob
import numpy as np
import cv2

"""**Dicionário RGB > Rótulo**"""
def gen_dict_cores():
    mask_dict = {
        (0, 0, 0): 0,        # Preto = None
        (100, 0, 100): 1,    # Violeta = Liver (fígado)
        (255, 255, 255): 2,  # Branco = Bone (Osso)
        (0, 255, 0): 3,      # Verde = Gallbladder (Vesícula biliar)
        (255, 255, 0): 4,    # Amarelo = Kidney (Rins)
        (0, 0, 255): 5,      # Azul = Pancreas
        (255, 0, 0): 6,      # Vermelho = Vessels (Veias)
        (255, 0, 255): 7,    # Rosa = Spleen (Baço)
        (0, 255, 255): 8     # Azul claro = Adrenal (Glândula Adrenal)
    }
    return mask_dict

""" Função que mapeia cada pixel da imagem ao valor rgb definido no dicionário,
 e substitui o valor pelo rótulo correspondente, em um novo array."""

def RGBtoClass(rgb, dictCores):
    arr = np.zeros(rgb.shape[:2])  # Inicializa a matriz de rótulos

    for color, label in dictCores.items():  # Itera sobre os pares (cor, rótulo)
        color = np.array(color)  # Converte a cor para um array NumPy
        arr[np.all(rgb == color, axis=-1)] = label  # Atribui o rótulo aos pixels que correspondem à cor

    return arr

def onehot_to_rgb(oneHot, dictCores):
    oneHot = np.array(oneHot)  # Converte para array numpy
    oneHot = np.argmax(oneHot, axis=-1)  # Seleciona o maior valor (índice)
    output = np.zeros(oneHot.shape + (3,))  # Cria a matriz RGB de saída
    oneHot = np.expand_dims(oneHot, axis=-1)  # Expande as dimensões

    for color, index in dictCores.items():
        output[np.all(oneHot == index, axis=-1)] = color

    return np.uint8(output)

"""esta função segmenta o nome do arquivo para o img_loader ordenar o dataset
na ordem do diretório e ter correspondência entre a lista de imagens e máscaras"""

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

""" img_loader recebe um caminho de diretório,uma lista vazia e  uma tupla com
dimensões de imagem. Lê as imagens png ou jpg do diretório, ordena pelo nome e
 armazena a imagem na lista img_data e seu nome na lista img_names"""
def img_loader(path, img_data, size=None, rgb=True):
  #lista para o nome dos arquivos
  img_names = []

  for diretorio_path in sorted(glob.glob(path)):
    for img_path in sorted(glob.glob(os.path.join(diretorio_path, "*.[pj]*[np]*[g]*")), key=natural_sort_key): #percorre o diretório na ordem natural dos títulos de arquivo
      img = cv2.imread(img_path,
                       cv2.IMREAD_COLOR if rgb
                       else cv2.IMREAD_GRAYSCALE) #img tem 3 canais na 3 dimensao se RGB, e 1 canal se preto/branco

      if rgb:  # Corrige para formato RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if size is not None:
        img = cv2.resize(img, size) #redimensiona conforme o parâmetro

      img_data.append(img.astype(np.uint8)) #add a imagem na lista do parametro
      img_names.append(os.path.basename(img_path)) #add o nome do arquivo na lista de nomes

  #return img_data, img_names
  return np.array(img_data), img_names

def save_dataset(X_train, X_val, X_test, y_train, y_val, y_test, save_dir="dataset"):
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    print(f"Dataset salvo em {save_dir}")

def load_dataset(save_dir="dataset"):

    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    X_val = np.load(os.path.join(save_dir, "X_val.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    y_val = np.load(os.path.join(save_dir, "y_val.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))

    print(f"Dataset carregado de {save_dir}")
    return X_train, X_val, X_test, y_train, y_val, y_test
