---
tags: #Documentação 
Título: 
Local: casa
Data: 14-03-2025
Subprojeto: 
Participantes: eu
---
Projeto: 
Referências:  

## Treinamentos e testes com imagens reais
---
Os testes  iniciais serão feitos usando apenas as 63 imagens e máscaras, combinando as funções de perda, métricas, uso de augmentation e Kfold. Depois, testarei juntar 63 imagens e máscaras reais com 63 imagens e máscaras sintéticas. 

Class Weights Calculados:
Classe 0: 0.19
Classe 1: 1.95
Classe 3: 142.67
Classe 4: 25.58
Classe 6: 38.23
Classe 7: 37.57

---
 nomeclatura: RUS_LOSS_versao_METODO_versao
 - a versão muda com a alteração de algum parâmetro, mas mantendo a função de perda e o método(k-fold, augmentation, augmentation+kfold)
### RUS_ACC01
para comparação, pois é o pior treinamento para o meu caso

lr = 0.003
metrics = ["accuracy", MeanIoU(num_classes=num_classes, name='mean_iou', sparse_y_true=True, sparse_y_pred=False)]
loss = keras.losses.SparseCategoricalCrossentropy()
augmentation: No

Callbacks
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',     # Monitora a perda de validação
    mode='min',             
    factor=0.3,             
    patience=3,             
    min_lr=10e-6,            
    min_delta=0.001,        
    verbose=1              
)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.9, patience=15, verbose=1, mode='auto')
checkpoint

history = model.fit(X_train, y_train,
                    batch_size=32,
                    verbose=1,
                    epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[lr_reducer, checkpointer, iou_callback],
                    shuffle=False)

Epoch 35 - Val Mean IoU: 0.2156
Classe 0: 0.8784
 | Classe 1: 0.1995
 | Classe 2: 1.0000
 | Classe 3: 0.0000
 | Classe 4: 0.0000
 | Classe 5: 1.0000
 | Classe 6: 1.0000
 | Classe 7: 0.0000
 | Classe 8: 1.0000
### RUS_ACC01_Au


input_shape = (256,256, 1)
num_classes = 9
lr = 0.003
metrics = ["accuracy", MeanIoU(num_classes=num_classes, name='mean_iou', sparse_y_true=True, sparse_y_pred=False)]
loss = keras.losses.SparseCategoricalCrossentropy()
augmentation: No

Callbacks
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',     # Monitora a perda de validação
    mode='min',             
    factor=0.3,             
    patience=3,             
    min_lr=10e-6,            
    min_delta=0.001,        
    verbose=1              
)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.9, patience=15, verbose=1, mode='auto')
checkpoint


Treinamento com augmentation ========================
  history = model.fit(augmenteixons(),
                      steps_per_epoch=len(X_train) // 4,
                      verbose=1,
                      epochs = 150,
                      validation_data=(X_val, y_val),
                      callbacks=[ checkpointer, iou_callback],
                      shuffle=False)

image_gen = ImageDataGenerator(
    rotation_range=25,            # Rotação aleatória d
    width_shift_range=0.2,        # Translação horizontal
    height_shift_range=0.2,       # Translação vertical
    shear_range=10,                # Cisalhamento
    zoom_range=[0.7, 1.3],
    fill_mode='constant',          # Preenchimento de regiões em branco após transformação
    
)


Epoch 45 - Val Mean IoU: 0.2155
Classe 0: 0.8803
 | Classe 1: 0.1974
 | Classe 2: 0.0000
 | Classe 3: 0.0000
 | Classe 4: 0.0000
 | Classe 5: 1.0000
 | Classe 6: 1.0000
 | Classe 7: 1.0000
 | Classe 8: 1.0000
10/10 ━━━━━━━━━━━━━━━━
### TVKY01
utilizando a tevrsky loss 01

**Parâmetros:**
beta = 0.5
alfa = 0.5
	essencialmente um dice_loss

lr = 1e-4
model_name = 'RUS_TVKY01'
testename = "RUS_TVK01"
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',     # Monitora a perda de validação
    mode='min',             
    factor=0.3,             
    patience=5,             
    min_lr=1e-6,            
    min_delta=0.001,        
    
    verbose=1              
)l

history = model.fit(X_train, y_train,
                    batch_size=4,
                    verbose=1,
                    epochs=300,
                    validation_data=(X_val, y_val),
                    callbacks=[lr_reducer, checkpointer, iou_callback],
                    shuffle=False)
### RUS_TVKY02

alterações:
lr = 5e-5
alfa = 0.7
beta = 0.3
	para penalizar mais falsos positivos
	
resultados:
o modelo está delimitando melhor os perímetros, mas com muitos falsos positivos e muito ruído.

### RUS_TVKY03

alterações:
batch_size = 2
epochs = 200
sem lr_reducer

### RUS_TVKY04

alterações:
alpha = 0.8
beta = 0.5
otimizador: AdamW(weight_decay=1e-4)
resultados significativamente melhores, pronto para testar com augmentation

### RUS_TVKY_FOCAL
combinação das funções Tversky Loss e Focal Loss
alpha=0.3,
beta=0.7, 
gamma=4/3


loss= 3e-5
history = model.fit(X_train, y_train,
                    batch_size=2,
                    verbose=1,
                    epochs=300,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpointer, iou_callback],
                    shuffle=False)
### RUS_TVKY_FOCAL02
alpha=0.7,
beta=0.7, 
gamma=1

loss= 5e-5
history = model.fit(X_train, y_train,
                    batch_size=2,
                    verbose=1,
                    epochs=300,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpointer, iou_callback],
                    shuffle=False)

Epoch 300 - Val Mean IoU: 0.2187
Classe 0: 0.8992
 | Classe 1: 0.4133
 | Classe 2: 1.0000
 | Classe 3: 0.0000
 | Classe 4: 0.0000
 | Classe 5: 1.0000
 | Classe 6: 0.0000
 | Classe 7: 1.0000
 | Classe 8: 0.0000
### TVKY_FOCAL02_Aug

alpha=0.3, beta=0.7, gamma=4/3

Epoch 130 - Val Mean IoU: 0.0488
Classe 0: 0.0856
 | Classe 1: 0.3397
 | Classe 2: 0.0000
 | Classe 3: 0.0043
 | Classe 4: 0.0093
 | Classe 5: 0.0000
 | Classe 6: 0.0000
 | Classe 7: 0.0000
 | Classe 8: 0.0000

### 'AUS_TVKY_FOCAL_GAN'
Dataset sintético convertido para Real utilizando CycleGans

alpha=1, beta=1, gamma=0.7 (jaccard + focal)

 ## Treinamento sem augmentation ========================
  history = model.fit(X_train, y_train,
                    batch_size=32,
                    verbose=1,
                    epochs=150,
                    validation_data=(X_val, y_val),
                    callbacks=[lr_reducer, checkpointer, iou_callback],
                    sample_weight=sample_weights,
                    shuffle=False)

### 'RUS_TVKY_FOCAL_DICE'

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',     # Monitora a perda de validação
    mode='min',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    min_delta=0.001,
    verbose=1
)

Epoch 243 - Val Mean IoU: 0.3641
Classe 0: 0.9197
 | Classe 1: 0.5369
 | Classe 2: 1.0000
 | Classe 3: 0.0000
 | Classe 4: 0.0000
 | Classe 5: 1.0000
 | Classe 6: 1.0000
 | Classe 7: 1.0000
 | Classe 8: 1.0000
	Mean IoU TESTE = 0.2707371
![[Pasted image 20250317150537.png]]

