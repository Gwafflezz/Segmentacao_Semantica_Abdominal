import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import save_dir
from model import Unet

##  Compilando ##############################3

# Definição da função de perda Jaccard
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    Jaccard Distance Loss é útil para datasets desbalanceados.
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
    sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


### Monitoramentto de IOU no treino
class IoUCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_argmax = np.argmax(y_pred, axis=-1).astype(np.int32)
        iou = MeanIoU(num_classes=num_classes)
        iou.update_state(self.validation_data[1], y_pred_argmax)
        print(f"Epoch {epoch + 1} - Mean IoU: {iou.result().numpy()}")


#parâmetros:
input_shape = (256,256, 1)
num_classes = 9
lr = 0.003
#metrics = ["accuracy",MeanIoU(num_classes=num_classes)]
metrics = ["accuracy"]
loss =  SparseCategoricalCrossentropy()
#loss = 'categorical_crossentropy'
#loss = jaccard_distance_loss(y_test, y_pred).numpy()

model = Unet(num_classes, input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
model.summary()


## Treinamento #####################################################


# Carregar os dados
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(save_dir="Data")

#y_train_cat = to_categorical(y_train, num_classes = 9)
#y_val_cat = to_categorical(y_val, num_classes = 9)

#y_test_cat = to_categorical(y_val, num_classes = 9)

## Data augmentation  ##########################################################

image_gen = ImageDataGenerator(
    rotation_range=15,            # Rotação aleatória de até 15 graus
    width_shift_range=0.3,        # Translação horizontal de até 30%
    height_shift_range=0.3,       # Translação vertical de até 30%
    shear_range=5,                # Cisalhamento de até 5%
    horizontal_flip=True,         # Flip horizontal
    vertical_flip=True,           # Flip vertical
    fill_mode='nearest',          # Preenchimento de regiões em branco após transformação
    brightness_range=[0.9, 1.1],  # Ajuste aleatório de brilho (80% a 120% do valor original)
    )

'''Mesmas transformações para as máscaras, exceto o brilho
para não afetar os valores das classes'''

mask_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

#ajustando os geradores para as imagens
image_gen.fit(X_train, augment=True)
mask_gen.fit(y_train, augment=True)

# Instanciando os geradores para mascaras e imagens
image_generator = image_gen.flow(X_train, batch_size=32, seed=42)
mask_generator = mask_gen.flow(y_train, batch_size=32
, seed=42)

# Combinando os geradores para gerar lotes com as mesmas transformações
train_generator = zip(image_generator, mask_generator)
def augmenteixons():
    for img, mask in train_generator:
        yield img, mask

## treinamento sem augmentation ################################################

#Parâmetros p/ callbacks
model_name = 'Jaccard01'
arquivo_modelo = f'Modelos/modelo_{model_name}.keras' # .h não é mais aceito
arquivo_modelo_json = f'Modelos/modelo_{model_name}.json'

#Callbacks
lr_reducer = ReduceLROnPlateau( monitor='val_loss', factor = 0.9, patience = 3, verbose = 1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience = 15, verbose = 1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor = 'val_loss', verbose = 1, save_best_only=True)


history = model.fit(X_train, y_train_cat,
                    batch_size=16,
                    verbose=1,
                    epochs = 100,
                    validation_data=(X_val, y_val_cat),
                    callbacks=[lr_reducer, early_stopper, checkpointer,IoUCallback()],
                    shuffle=False)

## Treinamento com augmentation ################################################
'''
model_name = 'Aug02'
arquivo_modelo = f'Modelos/modelo_{model_name}.keras' # .h não é mais aceito
arquivo_modelo_json = f'Modelos/modelo_{model_name}.json'


history = model.fit(augmenteixons(),
                    steps_per_epoch=len(X_train) // 32,
                    verbose=1,
                    epochs = 100,
                    validation_data=(X_val, y_val),
                    callbacks=[ checkpointer],
                    shuffle=False) '''

"""##Avaliação"""

_, acc = model.evaluate(X_test, y_test)
print("Accuracy is = ", (acc * 100.0), "%")

##Plota a loss e accuracy de treino e validação a acda época
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()