import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from utils import img_loader, save_dataset, RGBtoClass, gen_dict_cores
from sklearn.utils import class_weight

# caminhos para os diretórios das imagens

#real_img_treino = "/Data/abdominal_US/RUS/images/train"
# não tem mascara de treino pra ultrasons reais
sim_img_treino = "Data/abdominal_US/AUS/images/train"
sim_mask_treino = "Data/abdominal_US/AUS/annotations/train"

#real_img_val = "Data/abdominal_US/RUS/images/test"
#real_mask_val = "/Data/abdominal_US/RUS/annotations"
sim_img_val = "Data/abdominal_US/AUS/images/test"
sim_mask_val = "Data/abdominal_US/AUS/annotations/test"

#conjuntos de treino
  #imagens simuladas
Simg_treino = []
Simg_treino_names = []
Smask_treino = []
Smask_treino_names = []
  #imagens reais
"""Rimg_treino = []
Rimg_treino_names = []
#ñ tem mascara real p teste"""

#conjuntos de teste
  #imagens simuladas
Simg_val = []
Simg_val_names = []
Smask_val = []
Smask_val_names = []

  #imagens reais
"""Rimg_val = []
Rimg_val_names = []
Rmask_val = []
Rmask_val_names = []"""

#Carregando os dados:

dim = (256,256)
Simg_treino, Simg_treino_names = img_loader(sim_img_treino, Simg_treino,dim, False)
Smask_treino, Smask_treino_names = img_loader(sim_mask_treino,Smask_treino,dim)
#Rimg_treino, Rimg_treino_names = img_loader(real_img_treino,Rimg_treino,None,False)

Simg_val, Simg_val_names = img_loader(sim_img_val, Simg_val,dim, False)
Smask_val, Smask_val_names = img_loader(sim_mask_val,Smask_val,dim)
#Rimg_val, Rimg_val_names = img_loader(real_img_val, Rimg_val,None,False)
#Rmask_val, Rmask_val_names = img_loader(real_mask_val,Rmask_val)

'''concatenando as imagens e mascaras de treino / teste para separar ao carregar
no  train_test_split'''
all_imgs = np.concatenate((Simg_treino, Simg_val), axis=0)
all_masks = np.concatenate((Smask_treino, Smask_val), axis=0)

all_imgname = Simg_treino_names + Simg_val_names
all_maskname = Smask_treino_names + Smask_val_names

all_imgs  = np.expand_dims(all_imgs, axis=3)
#all_imgs.shape, all_masks.shape

"""**Convertendo máscaras**"""

dictCores = gen_dict_cores()
all_mask_class = []

for mask in  all_masks:
    onehotmask = RGBtoClass(mask, dictCores)
    all_mask_class.append(onehotmask)

all_mask_class = np.array(all_mask_class)
all_mask_class = np.expand_dims(all_mask_class, axis=3)

print("Classes únicas nos pixels das  máscaras :", np.unique(all_mask_class), all_mask_class.shape)

"""**Divisão do dataset em conjuntos de treino , validação e teste**


"""

# Divisão inicial: 10% para teste e 90% para treino e validação
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_imgs, all_mask_class, test_size=0.1, random_state=0, shuffle=False
)

# Segunda divisão: 15% para validação e 85% para teste
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=0, shuffle=False
)

# tamanhos dos conjuntos
print(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

#normalizando os conjuntos
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

# Salvando os dados
save_dataset(X_train, X_val, X_test, y_train, y_val, y_test, save_dir = "Data")

"""retorna X_train,

 pq o collab ta colapsando de tanta coisa na RAM
"""

""" 
class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
 """