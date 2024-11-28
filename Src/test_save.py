import numpy as np                      
import os                               
from keras.models import load_model    
from keras.metrics import MeanIoU      
from keras.utils import to_categorical  
from utils import gen_dict_cores, onehot_to_rgb, load_dataset
import cv2

model_name = 'Aug02'
model = load_model(f'Modelos/modelo_{model_name}.keras')
# Carregar os dados
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(save_dir="Data")

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#predição do conjunto de teste
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

####### avaliando pela metrica "intersection over union" no conjunto de teste ##
from keras.metrics import MeanIoU
n_classes = 9
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)

print("Mean IoU TESTE =", IOU_keras.result().numpy())

###### Salvando todas as predições ##################################

testename = "Teste5"
output_dir = f'Testes/{testename}'
os.makedirs(output_dir, exist_ok=True)  # Cria o diretório, se não existir
dictCores = gen_dict_cores()

# Itera sobre todo o conjunto de teste
for idx in range(len(X_test)):
    # Seleciona a imagem, a máscara e a predição correspondentes
    img = X_test[idx]
    mask = y_test[idx]
    pred = y_pred[idx]

    # Cria o plot
    plt.figure(figsize=(12, 4))

    # Imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Imagem Original")
    plt.axis('off')

    # Máscara
    plt.subplot(1, 3, 2)
    plt.imshow(onehot_to_rgb(to_categorical(mask, num_classes=9), dictCores))
    plt.title("Ground Truth")
    plt.axis('off')

    # Predição
    plt.subplot(1, 3, 3)
    plt.imshow(onehot_to_rgb(pred, dictCores), cmap='gray')
    plt.title("Predição")
    plt.axis('off')

    # Salva o plot no diretório
    output_path = os.path.join(output_dir, f"plot_{idx}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f"Todas as imagens foram salvas em: {output_dir}")
