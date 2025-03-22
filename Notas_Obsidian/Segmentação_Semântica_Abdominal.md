**Davi Bezerra Barros**

# 1. Introdução

A Ultrassonografia é uma técnica muito utilizada em diagnósticos medicinais, que aproveita o efeito doppler na reflexão de ondas sonoras de alta frequência - acima de 20kHz - para visualizar as estruturas internas de um organismo em tempo real. Por ser um método de baixo custo, não invasivo e indolor, a ultrassonografia é amplamente utilizada para avaliar uma grande variedade de órgãos do corpo humano, como:

- **Abdômen:** Fígado, vesícula biliar, pâncreas, baço, rins, bexiga e órgãos reprodutivos.
- **Pelve:** Útero, ovários, próstata e bexiga.
- **Mamas:** Para detectar nódulos e outros problemas.
- **Coração:** Para avaliar o funcionamento do coração e identificar doenças cardíacas.
- **Músculos e tendões:** Para diagnosticar lesões.
- **Vasos sanguíneos:** Para avaliar o fluxo sanguíneo e identificar obstruções.
- **Gestação:** Para acompanhar o desenvolvimento do feto, verificar a data provável do parto e identificar possíveis complicações.

![[k72.jpg|center]]

No entanto, a qualidade do exame depende da habilidade e experiência do profissional que o realiza, visto que eles precisam discriminar as propriedades ecogenéticas e características ruidosas inerentes aos tecidos observados. A qualidade da interpretação também está ligada à habilidade do radiologista em localizar áreas anatômicas utilizando o transdutor, evidenciando a necessidade de se criar métodos e ferramentas para o treinamento destes profissionais.

# 2. Metodologia

## 2.1. Objetivo do Projeto:
Para abordar estas dificuldades, este projeto propõe a implementação de um sistema capaz de identificar e destacar órgãos abdominais em imagens de ultrassonografia utilizando *segmentação semântica*, uma técnica de visão computacional. Para isso será utilizado um modelo de deep learning baseado na arquitetura **U-Net**, que se mostrou eficaz em tarefas de segmentação de imagens médicas. Métricas como ***IoU*(Intersection over Union)** e **Acurácia** serão utilizadas para avaliar o desempenho do modelo.

A inspiração e motivação para o desenvolvimento deste trabalho veio do artigo: "**Automated Deep Learning-Based Finger Joint Segmentation in 3-D Ultrasound Images With Limited Dataset**", disponível no link: https://www.researchgate.net/publication/384148850_Automated_Deep_Learning-Based_Finger_Joint_Segmentation_in_3-D_Ultrasound_Images_With_Limited_Dataset
## 2.2. Segmentação semântica

Segmentação de imagens, em visão computacional, é a tarefa de particionar uma imagem em várias regiões, onde os pixels dessas regiões devem compartilhar certas características. Há três grandes grupos de tarefas de segmentação de imagens: 

- Segmentação semântica
- Segmentação de instâncias
- Segmentação panóptica (semântica + instâncias)

O objetivo da segmentação semântica é categorizar *cada pixel* da imagem com a classe do objeto identificado. Isso é conhecido como *predição densa*, pois cada pixel da imagem está sendo inferido individualmente.

![[Pasted image 20241119230146.png|center|500]]

Como os objetos que pertencem a mesma classe não são instanciados individualmente, o mapa de segmentação não os identifica como objetos diferentes; **Esta é a tarefa da segmentação de instâncias**.

### 2.2.1. Funcionamento 

O objetivo da segmentação semântica, em essência, é receber como input uma imagem RGB (H, W, 3) ou em escala de cinza (H, W, 1) e produzir um mapa de segmentação onde cada pixel é um valor inteiro que representa uma classe (H, W, 1). 

![[Pasted image 20241120022944.png|center]]

Como esta é uma tarefa *categórica*, tanto as máscaras de treinamento quanto as máscaras previstas são representadas por vetores one-hot-encoded com formato (H, W, C) onde C é o número de canais que representam as probabilidades das classes. Ao selecionar a maior probabilidade para cada pixel, o mapa retorna ao formato (H, W, 1) e o resultado pode ser observado ao sobrepor a imagem original:

![[Pasted image 20241120023929.png|center]]

## 2.3. Arquitetura da Rede

A arquitetura de rede escolhida para o projeto foi a **U-net**, uma rede totalmente convolucional proposta em 2015, especificamente para tarefas de segmentação de imagens biomédicas, mas que teve seu uso expandido para outras tarefas de segmentação devido ao à sua eficiência e simplicidade. Sua estrutura em formato de "U"(daí o nome) foi criada de forma a permitir o treinamento com poucas imagens, ao utilizar métodos de data augmentation. Esta capacidade a torna ideal para a aplicação no campo da biomedicina, onde muitas vezes não é possível coletar uma grande amostra de imagens para treinamento.


![[Pasted image 20241120182648.png|center]]

### 2.3.1. Estrutura

A U-net é composta por dois componentes principais: um caminho de *contração*(encoder) e um caminho de *expansão*(decoder), conectados por *skip connections* para permitir a localização exata das características extraídas. 

#### 2.3.1.1. Caminho de contração:
O caminho de contração captura informações contextuais e características globais da imagem, enquanto reduz as suas dimensões e aumenta o número de mapas de características. Cada bloco de contração é composto por:

- **Convolução:** Duas camadas de convolução 3x3 ativadas por ReLU.
- **Downsampling:** Redução das dimensões com uma camada de *Max Pooling* 2x2.

![[Pasted image 20241120181802.png|center|500]]

O caminho de contração é formado por 5 blocos convolucionais que extraem as características locais, seguidos por uma camada de *max pooling* para reduzir a dimensionalidade espacial enquanto mantém as características relevantes. A cada bloco o número de filtros é dobrado para capturar informações mais abstratas nas camadas mais profundas.

**Bloco 1:**
1. Recebe a imagem de entrada de formato (572, 572, 1) que será processada pela U-net.
2. Duas camadas de convolução 3x3 e padding 'valid' são aplicadas à imagem, seguidas por uma camada de ativação ReLU, gerando **64** mapas de características.
3. Em seguida, é aplicada uma camada de *Max Pooling 2x2* reduzindo os mapas de características à metade de suas dimensões.

**Bloco 2:**
1. Recebe a imagem do bloco anterior
2. Aplica as mesmas camadas convolucionais, dobrando o número de mapas para **128.**
3. Max pooling 2x2 reduzindo à metade as dimensões do mapa ao sair da segunda camada convolucional.

**Bloco 3:**  Mesmo processo dos blocos 1 e 2, gerando **256** mapas de características.
**Bloco 4:** Mesmo processo dos blocos 1, 2 e 3, gerando **512** mapas de características.

**Bloco de conexão (Bottleneck):* *
1. Recebe os mapas da camada 4 e aplica 2 camadas de convolução 3x3, gerando **1024** mapas de característica.
2. Fornece a transição dos mapas gerados no caminho de contração, para o caminho de expansão.

#### 2.3.1.2. Caminho de expansão:

O caminho de expansão reconstrói a imagem, mapeando as características aprendidas de volta para a resolução original. Ele utiliza técnicas de  *upsampling* e concatenações (*skip connections*) com os mapas de características dos blocos da camada de contração para preservar a localização das informações extraídas. Cada bloco é composto por: 

- **Upsampling:** Aumento da dimensão dos mapas de características utilizando convoluções transpostas.
- **Concatenação:** Adição dos mapas de características do bloco simétrico no caminho de contração.
- **Convolução:** Duas camadas convolucionais 3x3 com ativação ReLU.


![[Pasted image 20241120185657.png|center|500]]



**Bloco 6:**
1. Recebe os mapas comprimidos do bottleneck com 1024 mapas de características.
2. Realiza um _upsampling_ para dobrar as dimensões espaciais, reduzindo o número de mapas de características para **512.**
3. Concatena os mapas gerados com os mapas correspondentes do Bloco 4, gerando 1024 mapas.
4. Aplica duas convoluções 3×3 seguidas por ReLU, reduzindo os mapas para 512.

**Bloco 7:** Mesmo processo do bloco 6, reduz os mapas de características para **256**.

**Bloco 8:** Mesmo processo do bloco 7, reduz os mapas de características para **128**.

**Bloco 9(Saída):**
1. Recebe os mapas do Bloco 8 e realiza um upsampling, dobrando as dimensões espaciais e reduzindo os mapas para 64.
2. Concatena com os mapas correspondentes do bloco 1, gerando 128 mapas.
3. Aplica uma convolução 3x3 seguidas por ReLU, reduzindo os mapas para 64.
4. Aplica uma convolução 1x1 seguida por uma camada de ativação *Sigmoide*, reduzindo o número de canais de características para a quantidade de classes desejada. Gera o mapa de segmentação final, no mesmo tamanho da imagem original, com cada pixel representando uma classe no formato de one-hot encoding.

Essa arquitetura garante que tanto as informações contextuais geradas pelo encoder quanto os detalhes espaciais das características sejam aproveitados.
# 3. Implementação do projeto

A seguir serão descritas e demonstradas todas as etapas da implementação deste projeto, desde o pré-processamento dos dados até a visualização e avaliação dos resultados. Todo o código e comentários de desenvolvimento podem ser encontrados no notebook: https://colab.research.google.com/drive/1drz8sJ3-XT-IUotxwml6d1YeRC2AHLE0?usp=sharing
## 3.1. Aquisição de dados

O dataset escolhido para este trabalho foi o "US simulation & segmentation", disponível no kaggle pelo link: https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm

Este dataset foi desenvolvido com  imagens reais de ultrassonografia e imagens sintéticas geradas por um simulador baseado em ray-casting, a partir de dados volumétricos de **Tomografia Computadorizada**. As imagens representam órgãos abdominais, como fígado, pâncreas, adrenais, vesícula biliar, baço, vasos sanguíneos e ossos, totalizando 694 imagens anotadas divididas entre os conjuntos reais e sintéticos.

**Conteúdo**
O dataset está dividido entre imagens reais e imagens sintéticas, já separados em subconjuntos de treino e teste.

1. **Imagens Reais de Ultrassonografia:** Total de 617 imagens
   ![[Pasted image 20241120211024.png|center]]
	1. **Treino:
		- 404  imagens reais de ultrassonografia.
		- Nenhuma imagem possui anotação no subconjunto de treino.
	2. **Teste:**
		 - 213 imagens reais de ultrassonografia.
		 - 61 máscaras de segmentação anotadas manualmente.

2. **Imagens Sintéticas de Ultrassonografia:** Total de 926 imagens
   ![[Pasted image 20241120211424.png|center]]
	1. **Treino:**
		-  633 imagens sintéticas de ultrassonografia.
	    - 633 máscaras de segmentação
    1. **Teste:**
		-  293 imagens sintéticas de ultrassonografia.
		- 293 máscaras de segmentação anotadas automaticamente.

**Anotações**
As anotações são representadas por cores que identificam diferentes órgãos abdominais:

- **Violeta:** Fígado.
- **Amarelo:** Rins.
- **Azul:** Pâncreas.
- **Vermelho:** Vasos.
- **Azul claro:** Adrenais.
- **Verde:** Vesícula biliar.
- **Branco:** Ossos.
- **Rosa:** Baço.
## 3.2. Implementação da U-Net

A arquitetura da rede foi implementada manualmente seguindo o artigo original "U-Net: Convolutional Networks for Biomedical Image Segmentation", disponível no link: https://arxiv.org/pdf/1505.04597v1 

Para a modelagem do modelo, foi utilizado o framework **TensorFlow** com a **API Keras** da seguinte maneira: 

**importação dos pacotes**

```python
import tensorflow as tf
from tensorflow.keras import  Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import SparseCategoricalCrossentropy

```

### 3.2.1. Funções e arquitetura

**Bloco convolucional:** Função que encapsula operações repetidas de convolução 3×3, Batch Normalization, ativação ReLU e dropout.

```python
def ConvBlock(tensor, num_feature):

	x = tf.keras.layers.Conv2D(num_feature,
										(3,3),
										activation='relu',
										kernel_initializer='he_normal',
										padding='same')(tensor)
	
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(0.1)(x)
	x = tf.keras.layers.Conv2D(num_feature,
										3,3),
										activation='relu',
										kernel_initializer='he_normal',
										padding='same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	return x
```
- Configura os pesos usando inicialização HeNormal.
<br>

**Bloco "Bottleneck":** 
```python
def ponte(tensor, num_feature):
  x = ConvBlock(tensor,num_feature)
  return x
```
- Conecta as seções encoder e decoder.
- Aplica um bloco convolucional com o maior número de filtros.
<br>

**Blocos encoder e decoder :** Aplicam as convoluções, downsamplings, upsamplings, concatenações e passam o tensor adiante.
```python
#Seção encoder
def encoder_block(tensor, num_feature):
  x = ConvBlock(tensor, num_feature)
  p = tf.keras.layers.MaxPooling2D((2,2))(x)
  return x, p

#seção decoder
def decoder_block(tensor,skip_connection, num_feature):
  x = tf.keras.layers.Conv2DTranspose(num_feature, (2,2), strides=(2,2), padding='same')(tensor) #recebe do bloco anterior e faz upsampling
  x = tf.keras.layers.concatenate([x, skip_connection])
  x = ConvBlock(x, num_feature)
  return x
```
**Seção encoder:**
- Aplica o _ConvBlock_ para extração de características.
- Utiliza _MaxPooling_ para reduzir a dimensão da saída.
- Armazena o tensor de saída sem _pooling_ para ser utilizado nas _skip connections_.

**Seção decoder:**
- Realiza _upsampling_ com convoluções transpostas.
- Concatenada com as _skip connections_ do encoder.
- Refinada por um bloco convolucional.
<br>

**Definição da arquitetura:** Pipeline dos dados ao longo de toda a estrutura
```python
def Unet(n_classes, tensor_shape):

  input = tf.keras.layers.Input(tensor_shape) #instancia o tensor para os dados de entrada

  #seção de contração:
  skip1, c1 = encoder_block(input,16) # 128x128x3 > 64x64x16
  skip2, c2 = encoder_block(c1,32) # 64x64x16 > 32x32x32
  skip3, c3 = encoder_block(c2,64) # 32x32x32 > 16x16x64
  skip4, c4 = encoder_block(c3,128) # 16x16x64 > 8x8x64

  #bottleneck
  c5 = ponte(c4,256) # 8x8x64 > 8x8x256

  #seção de expansão:
  c6 = decoder_block(c5, skip4, 128) #8x8x256 > 16x16x128
  c7 = decoder_block(c6, skip3, 64) #16x16x128 > 32x32x64
  c8 = decoder_block(c7, skip2, 32) #32x32x64 > 64x64x32
  c9 = decoder_block(c8, skip1, 16) #64x64x32 > 128x128x16

  #camada de saída:
  output = tf.keras.layers.Conv2D(n_classes, (1,1), activation='softmax')(c9) #128x128x16 > 128x128x8, 8= número de classes

  model = Model(input, output, name="U-Net")
  return model
```
- A arquitetura foi construída combinando os blocos encoder, ponte e decoder, realizando as *skip connections*.
- A camada de saída foi adicionada com uma ativação softmax para prever as classes dos pixels.
<br>

**Compilação:** 

```python
#parâmetros:
input_shape = (256,256, 1)
classes = 9
lr = 1e-3
metrics = ["accuracy"]
loss =  SparseCategoricalCrossentropy()

model = Unet(classes, input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
model.summary()

#Arquivos par asalvamento do modelo
arquivo_modelo = '/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/modelo_01_UnetUS.keras' # .h não é mais aceito
arquivo_modelo_json = '/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/modelo_01_UnetUS.json'

#Callbacks
lr_reducer = ReduceLROnPlateau( monitor='val_loss', factor = 0.9, patience = 3, verbose = 1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience = 15, verbose = 1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor = 'val_loss', verbose = 1, save_best_only=True)
```
**Funções de Perda e Métricas:** 
- `SparseCategoricalCrossentropy` como função de perda.
- Precisão como métrica de aprendizado

**Callbacks:**
- `ReduceLROnPlateau`: Para reduzir a taxa de aprendizado automaticamente.
- `EarlyStopping`: Para interromper o treinamento caso não haja melhora após um certo número de épocas.
- `ModelCheckpoint`: Para salvar o melhor modelo durante o treinamento.
<br>

**Verificando se a saída do modelo está do formato esperado:**

```python
#testando o modelo
import numpy as np

input_image = np.random.rand(1, 256, 256, 1)
output_image = model.predict(input_image)

input_shape = input_image.shape
output_shape = output_image.shape

print("Dimensão da entrada:", input_shape)
print("Dimensão da saída:", output_shape)
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 996ms/step
Dimensão da entrada: (1, 256, 256, 1)
Dimensão da saída: (1, 256, 256, 9)

A dimensão de entrada foi reduzida para 256x256 para poupar memória.

## 3.3. Pré-processamento
O pré-processamento foi projetado de forma a permitir que seu pipeline seja aplicado a qualquer conjunto de imagens. 
### 3.3.1. Carregamento das imagens

**Funções:**
```python
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
```

- **`natural_sort_key(s)`**:
    - Essa função ordena os arquivos de forma natural, tratando números contidos nos nomes dos arquivos para garantir que a ordem seja adequada e garantir a correspondência correta entre imagens e máscaras.
- **`img_loader(path, img_data, size=None, rgb=True)`**:
    - Carrega as imagens de um diretório, redimensiona para as dimensões especificadas (`size`), e converte as imagens para o formato RGB (se necessário).
    - As imagens são armazenadas na lista `img_data` e os nomes dos arquivos são armazenados em `img_names`.

### 3.3.2. Processamento de Máscaras

#### 3.3.2.1. Funções
Funções utilizadas para manipular as mascaras alimentadas ao modelo e geradas por ele.

**Dicionário de Cores:**
```python
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
```
 Cria um dicionário que mapeia cores RGB para rótulos numéricos. Cada cor representa uma estrutura anatômica diferente, como fígado, osso, vesícula biliar, etc.


**Conversor de matriz RGB para Matriz de rótulos:** Essa função converte uma matriz RGB para uma matriz de inteiros .

```python
def RGBtoClass(rgb, dictCores):
    arr = np.zeros(rgb.shape[:2])  # Inicializa a matriz de rótulos

    for color, label in dictCores.items():  # Itera sobre os pares (cor, rótulo)
        color = np.array(color)  # Converte a cor para um array NumPy
        arr[np.all(rgb == color, axis=-1)] = label  # Atribui o rótulo aos pixels que correspondem à cor
    return arr
```

Para cada pixel da imagem de máscara, a cor é mapeada para o rótulo correspondente, substituindo a cor pelo valor do rótulo. A função utiliza uma operação matricial para a conversão, essencialmente fazendo uma comparação booleana AND pixel-a-pixel. Isso faz com que o processo tenha um menor custo computacional.

<br>
**Conversor de matriz de vetores One-Hot para RGB:** Essa função converte uma mascara do formato one-hot-encoded de volta para RGB. 

```python
def onehot_to_rgb(oneHot, dictCores):
    oneHot = np.array(oneHot)  # Converte para array numpy
    oneHot = np.argmax(oneHot, axis=-1)  # Seleciona o maior valor (índice)
    output = np.zeros(oneHot.shape + (3,))  # Cria a matriz RGB de saída
    oneHot = np.expand_dims(oneHot, axis=-1)  # Expande as dimensões

    for color, index in dictCores.items():
        output[np.all(oneHot == index, axis=-1)] = color

    return np.uint8(output)
```

A saída de um modelo de segmentação geralmente é da forma (batch_size, n_classes, altura, largura). Cada índice no vetor one-hot é mapeado para a cor correspondente definida no dicionário de cores.

#### 3.3.2.2. Testando as conversões:
Foi feito um teste específico para cada uma das funções, utilizando imagens riadas manualmente:

**Teste RGB > Matriz de rótulos:**

```python
# Criando uma imagem RGB 3x3 com as cores do dicionário
rgb_image = np.array([
    [(255, 255, 255), (100, 0, 100), (0, 255, 0)],
    [(0, 0, 255), (255, 0, 0), (255, 0, 255)],
    [(0, 255, 255), (255, 255, 0), (0, 0, 0)]
])

dictCores = gen_dict_cores()
# gerando matriz de rotulos]
one_hot_image = RGBtoClass(rgb_image, dictCores)

# Exibindo a imagem RGB gerada
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

# Exibindo a matriz de rótulos 
print("Matriz de rótulos:")
print(one_hot_image)
```
![[Pasted image 20241121010505.png|250]]
Matriz de rótulos (one-hot): 
[ [2. 1. 3.]
..[5. 6. 7.] 
..[8. 4. 0.]]
O resultado está de acordo com o dicionário e com a posição das cores.

**Teste Matriz One-Hot > RGB:**

```python
# Definindo manualmente as classes para cada pixel da imagem 3x3
oneHot = np.zeros((3, 3, 9))  # Imagem 3x3 com 9 classes

# One-hot com as classes do dicionário:

#esse resultado explicito quem o faz é o argmax
oneHot[0, 0, 2] = 1  # Branco (Osso) (0,0)
oneHot[0, 1, 5] = 1  # Azul (Pancreas) (0,1)
oneHot[0, 2, 3] = 1  # Verde (Vesícula) (0,2)

oneHot[1, 0, 1] = 1  # Violeta (Fígado) (1,0)
oneHot[1, 1, 6] = 1  # Vermelho (Veias) (1,1)
oneHot[1, 2, 7] = 1  # Rosa (Baço) (1,2)

oneHot[2, 0, 0] = 1  # Preto (Nada) (2,0)
oneHot[2, 1, 2] = 1  # Branco (Osso) (2,1)
oneHot[2, 2, 8] = 1  # Azul claro (Gland.Adrenal) (2,2)

# Gerando o dicionário
dictCores = gen_dict_cores()

# Convertendo o vetor one-hot para a imagem RGB
rgb_image = onehot_to_rgb(oneHot, dictCores)

plt.imshow(rgb_image)
plt.axis('off')
plt.show()
print("Dimensão da saída do modelo:", oneHot.shape)
print("Dimensão da imagem RGB gerada:", rgb_image.shape)
```
![[Pasted image 20241121011106.png|250]]
Dimensão da saída do modelo: (3, 3, 9)
Dimensão da imagem RGB gerada: (3, 3, 3)

Aqui foi definido uma matriz 3x3x1 manualmente, onde cada posição tem o valor de um rótulo definido no dicionário. O modelo gera probabilidades para as dimensões do vetor, mas como a função **argmax** colapsa o vetor para o maior valor, a representação das probabilidades foi deixada para o teste seguinte.

**Teste 2 One-hot > RGB:**
```python
output_image = np.random.rand(1, 4, 4, 9)  # Imagem de saída com 9 classes
output_image = np.squeeze(output_image, axis=0)  # Remove a dimensão do batch(so tem uma img)

# Criando o dicionário
dictCores = gen_dict_cores()

# Convertendo a saída do modelo para a imagem RGB
rgb_image = onehot_to_rgb(output_image, dictCores)

# Imagem gerada
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

print("Dimensão da saída do modelo:", output_image.shape)
print("Dimensão da imagem RGB gerada:", rgb_image.shape)
print(output_image)
```
![[Pasted image 20241121011727.png|250]]
Dimensão da saída do modelo: (4, 4, 9) 
Dimensão da imagem RGB gerada: (4, 4, 3)
```python
[[[0.57908881 0.33580392 0.34410017 0.23720563 0.13586635 0.05622759
   0.5862357  0.96747813 0.45280148]
  [0.8930983  0.18276845 0.62709657 0.07603959 0.67300906 0.67327464
   0.61237694 0.21165752 0.22208691]
  [0.22039128 0.94177474 0.33338415 0.51746079 0.59490881 0.01584007
   0.66245684 0.42296073 0.67942008]
  [0.93072832 0.47512061 0.56555351 0.00935921 0.76246471 0.24776436
   0.67333821 0.60336891 0.57013791]]

 [[0.67547263 0.97666166 0.48209579 0.05390186 0.98060942 0.56406069
   0.31010073 0.61860062 0.00416096]
  [0.67526767 0.77943618 0.5162135  0.08672409 0.32712662 0.58154258
   0.67918167 0.38743725 0.85377961]
  [0.10333022 0.33368111 0.2277703  0.36395461 0.17047299 0.31021793
   0.13229725 0.23419578 0.77836624]
  [0.93419304 0.82677624 0.41762912 0.72430383 0.60024323 0.73791163
   0.30090767 0.87661449 0.28460251]]

 [[0.54004565 0.31069832 0.91982634 0.50408192 0.86804815 0.66417395
   0.8121413  0.37085723 0.24864316]
  [0.47253666 0.74654554 0.94921976 0.65585023 0.01949505 0.69139072
   0.69685657 0.83856082 0.2469597 ]
  [0.89189944 0.0755049  0.35849327 0.60262315 0.28713629 0.91245026
   0.92043945 0.47336987 0.96560221]
  [0.89639344 0.1702278  0.23391221 0.46912105 0.98255504 0.47398564
   0.29338927 0.39266837 0.42605398]]

 [[0.54492246 0.48854141 0.63795983 0.95875354 0.46468765 0.97960605
   0.65112426 0.52646422 0.9889281 ]
  [0.23461664 0.77465378 0.12610709 0.07643158 0.60316224 0.59314858
   0.81460461 0.05123225 0.89071958]
  [0.53553578 0.82616564 0.98435011 0.70290783 0.75915189 0.8779408
   0.21457065 0.38440187 0.11100548]
  [0.57730195 0.68633509 0.7877986  0.79553713 0.85351347 0.06763191
   0.27510914 0.62280713 0.78067767]]]
```
Chatinho de ler, mas é possível verificar. A função funciona muito bem.

### 3.3.3. Separação dos conjuntos
Com as funções de conversão definidas para as máscaras, agora é possível aplicar as transformações necessárias às imagens para montar os conjuntos de treino, validação e teste. O fluxograma abaixo ilustra todas as etapas do processamento necessário para preparar os dados e fornecê-los ao modelo:

![[PreProcessamento_Fluxograma4.png]]

**Carregando os dados**

 As imagens e as máscaras são armazenadas em arrays `Simg_treino`, `Smask_treino`, `Simg_val`, e `Smask_val` com a função img_loader. As imagens reais não serão utilizadas na primeira etapa do treinamento.

```python
#Carregando os dados:
dim = (256,256)
Simg_treino, Simg_treino_names = img_loader(sim_img_treino, Simg_treino,dim, False)
Smask_treino, Smask_treino_names = img_loader(sim_mask_treino,Smask_treino,dim)

Simg_val, Simg_val_names = img_loader(sim_img_val, Simg_val,dim, False)
Smask_val, Smask_val_names = img_loader(sim_mask_val,Smask_val,dim)
```

Após o carregamento os dados são plotados para verificar a correspondência entre  as imagens e máscaras:
![[Pasted image 20241121003533.png|center|500]]
As listas de nomes permitiram a identificação correta de cada arquivo e da não correspondência entre as máscaras e imagens. A função *natural_sort_keys()* foi criada em função desse problema, pois os arquivos estavam sendo carregados fora de ordem.

**Concatenando o dataset real e simulado**

O dataset já veio separado com os subconjuntos de treino e teste que o criador utilizou, mas preferi concatenar tudo para utilizar uma divisão diferente.
```python
'''concatenando as imagens e mascaras de treino / teste para separar ao carregar
no  train_test_split'''
all_imgs = np.concatenate((Simg_treino, Simg_val), axis=0)
all_masks = np.concatenate((Smask_treino, Smask_val), axis=0)

all_imgname = Simg_treino_names + Simg_val_names
all_maskname = Smask_treino_names + Smask_val_names

all_imgs  = np.expand_dims(all_imgs, axis=3)
all_imgs.shape, all_masks.shape
```

Mais uma verificação para garantir que a correspondência entre imagens e máscaras não foi alterada:
![[Pasted image 20241121163739.png|center| 500]]

**Convertendo todas as máscaras de RGB para mapas de classes:**

```python
dictCores = gen_dict_cores()
#all_img_class = [] #as imagens não precisam ser convertidas para índices de classe/pixel
all_mask_class = []

"""for image in  all_imgs:
    #img_all = np.expand_dims(img_all, axis=3)
    onehotimage = RGBtoOneHot(image, dictCores)
    all_img_class.append(onehotimage)"""

for mask in  all_masks:
    onehotmask = RGBtoClass(mask, dictCores)
    all_mask_class.append(onehotmask)

#convertendo de volta para np.array
#train_img = np.array(all_img_class)
all_mask_class = np.array(all_mask_class)

#expandindo 1 dimensão
#train_img_all = np.expand_dims(all_img_class, axis=3)
all_mask_class = np.expand_dims(all_mask_class, axis=3)

#print("Classes únicas nos pixels das imagens :", np.unique(all_img_class))
print("Classes únicas nos pixels das  máscaras :", np.unique(all_mask_class), all_mask_class.shape)
#print("Formato das imagens :", all_img_class.shape)
```
 `Classes únicas nos pixels das máscaras : [0. 1. 2. 3. 4. 5. 6. 7. 8.] (926, 256, 256, 1)`
  
Ao final foi feita uma verificação dos valores únicos dos pixels e mtodo o dataset com a função **np.unique**. Isso confirma que as conversões foram bem sucedidas.

Mais uma verificação, utilizando a função *onehot_to_rgb* para visualizar as máscaras convertidas:
![[Pasted image 20241121164348.png|center|500]]
`((256, 256, 1), (256, 256, 3), (256, 256, 1), (256, 256, 3))`

Isso confirma o funcionamento das duas funções, *RGBtoClass* e *onehot_to_rgb*.

**Divisão em conjuntos de treino e validação:**
O conjunto de imagens `all_imgs`  e de máscaras `all_mask_class` foram divididos entre conjuntos de treino, validação e teste..

```python 
# Divisão inicial: 10% para teste e 90% para treino e validação
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_imgs, all_mask_class, test_size=0.1, random_state=0, shuffle=False
)

# Segunda divisão: 15% para validação e 85% para teste
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=0, shuffle=False
)

# tamanhos dos conjuntos
print(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")```

**Normalização dos dados:**
Modelos de deep learning trabalham melhor com valores dos dados entre 0 e 1. Os dados foram normalizados para auxiliar na convergência da rede.
```python
#normalizando as imagens
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)
```

Mais um teste do dataset - por que não? - Verificando todas as características dos dados:
```python
#Teste de sanidade
print("Classes únicas em y_train", np.unique(y_train), y_train.shape, y_train.dtype)
print("Classes únicas em y_val", np.unique(y_val), y_val.shape, y_val.dtype)
print("Classes únicas em y_test:", np.unique(y_test), y_test.shape, y_test.dtype)

print("Formato de X_train:", X_train.shape, X_train.dtype)
print("Formato de X_val:", X_val.shape, X_val.dtype)
print("Formato de X_test", X_test.shape, X_test.dtype)
```
`Classes únicas em y_train [0 1 2 3 4 5 6 7 8] (833, 256, 256, 1) int32` 
`Classes únicas em y_test: [0 1 2 3 4 5 6 7 8] (93, 256, 256, 1) int32` 
`Formato de X_train: (833, 256, 256, 1) float32` 
`Formato de X_test (93, 256, 256, 1) float32`

E outra visualização do dataset dividido, normalizado e com os tipos convertidos:
![[Pasted image 20241121165904.png|center|500]]

As máscaras foram convertidos para inteiros na tentativa de implementar um balanceamento no dataset, embora não tenha sido implementado até o momento(não consegui resolver o erro):
```python
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train.flatten())  # Flatten y_train to 1D

print("Os pesos das classes são:", class_weights)
#dicionário com pesos das classes para balanceamento
class_weights = {
    class_weights[0]: 0,
    class_weights[1]: 1,
    class_weights[2]: 2,
    class_weights[3]: 3,
    class_weights[4]: 4,
    class_weights[5]: 5,
    class_weights[6]: 6,
    class_weights[7]: 7,
    class_weights[8]: 8
}
```
`Os pesos das classes são: [1.33613583e-01 9.49245410e-01 4.97438956e+01 2.72413430e+01 6.13989077e+00 1.34157296e+01 3.94479946e+01 7.19208821e+00 2.73934015e+02]`
A classe 02 (fígado, roxo) era a classe predominante depois do preto, e seu peso foi significativamente reduzido em relação aos demais.

**Funções para salvamento**
Foi necessário salvar dataset dividido (como np.array) e evitar processar tudo novamente (o colab estava quebrando por causa da memória T.T ):

```python
def save_dataset(X_train, X_test, y_train, y_test, save_dir="dataset"):

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    print(f"Dataset salvo em {save_dir}")
    
def load_dataset(save_dir="dataset"):
 X_train = np.load(os.path.join(save_dir, "X_train.npy"))
  X_test = np.load(os.path.join(save_dir, "X_test.npy"))
  y_train = np.load(os.path.join(save_dir, "y_train.npy"))
  y_test = np.load(os.path.join(save_dir, "y_test.npy"))

  print(f"Dataset carregado de {save_dir}")
  return X_train, X_test, y_train, y_test
```

Executando o salvamento:
```python
save_dataset(X_train, X_test, y_train, y_test, save_dir = "/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal")
```

Além disso, seguindo a recomendação de um colega para verificar se as imagens estão ok depois de todo o processo, uma materialização das imagens em si:

```python
import imageio
def salvar_imagens_em_diretorio(data, diretorio_destino):
    """
    Salva as imagens de um array NumPy em um diretório específico.
    """
    # Criar o diretório se ele não existir
    os.makedirs(diretorio_destino, exist_ok=True)

    # Salvar as imagens
    for i in range(data.shape[0]):
        nome_arquivo = f"imagem_{i}.jpg"
        caminho_completo = os.path.join(diretorio_destino, nome_arquivo)
        imageio.imwrite(caminho_completo,data[i])
        #imageio.imwrite(caminho_completo, onehot_to_rgb(data[i],dictCores))
```

```python
salvar_imagens_em_diretorio(X_train, "/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal/X_train")
salvar_imagens_em_diretorio(X_test, "/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal/X_test")
salvar_imagens_em_diretorio(y_train, "/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal/y_train")
salvar_imagens_em_diretorio(y_test, "/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal/y_test")
```
## 4. Treinamento

**Carregando o dataset processado:**
O primeiro passo para iniciar o treinamento, é carregar o dataset como np.array:
```python
#carregando os conjuntos de dados
X_train, X_test, y_train, y_test = load_dataset(save_dir="/content/drive/MyDrive/Colab Notebooks/Segementação_Abdominal/Dataset_UltrassomAbdominal")
```

**Conversão das máscaras para one-hot:**
A função **to_categorical** converterte os índices de classe inteiros para  vetores one-hot, transformando as máscaras para o formato (h, w, 9).
```python 
y_train_cat = to_categorical(y_train, num_classes = 9)
y_test_cat = to_categorical(y_test, num_classes = 9)
```
Do segundo teste em diante  esta conversão não foi mais utilizada, para poupar o uso de memória da plataforma. Por conta disso a função de perda do modelo mudou de *categorical_cross_entropy* para *sparse_categorical_cross_entropy*, que funciona com o formato (h, w, 1).

**Treinamento**

```python
history = model.fit(X_train, y_train,
                    batch_size=16,
                    verbose=1,
                    epochs = 50,
                    validation_data=(X_test, y_test),
                    callbacks=[lr_reducer, early_stopper, checkpointer],
                    shuffle=False)
```

Para o treinamento foi utilizado um batch_size de 16, 50 épocas e os callbacks definidos anteriormente. O treinamento com balanceamento do dataset não funcionou. O treinamento durou apenas 16 épocas, e logo foi encerrado pelo ***early_stopping***.
## 5. Resultados e Discussão:
Nesta seção serão descritos os testes, avaliações e métricas de avaliação utilizadas para analisar o desempenho do modelo de segmentação U-Net aplicado nas imagens de ultrassom.
### 5.1. Métricas de avaliação
Para a avaliação do modelo, foram utilizadas as seguintes métricas:
#### 5.1.1. Acurácia
A acurácia é a métrica mais comum, indicando a porcentagem de pixels corretamente classificados. Em problemas de segmentação ela não é a mais indicada, especialmente para datasets desbalanceados pois ela será enviesada pelos **verdadeiros negativos**.

![[Pasted image 20241122124533.png|center]]

![[Pasted image 20241122124947.png|center]]

- **TP**: Pixels corretamente classificados.
- **TN**: Pixels corretamente não classificados
- **FP**: Pixels incorretamente classificados.
- **FN**: Pixels que foram ignorados
#### 5.1.2. IoU(Intersection Over Union) ou Jaccard Coefficient
A **IoU** é a métrica mais adequada para avaliar modelos de segmentação, especialmente os multiclasse. Ela calcula o percentual da predição que coincide com o **Ground Truth**. A interseção (𝐴∩𝐵) são os pixels que pertencem à predição e à máscara ground truth, e a união (𝐴∪𝐵) são todos os pixels contidos nos dois.


![[Pasted image 20241122131328.png|center]]

**Exemplo**
Ground truth e predição:
![[Pasted image 20241122125247.png]]
Intersecção e União:
![[Pasted image 20241122125253.png]]


Pixel-a-Pixel:
![[Pasted image 20241122124700.png|center]]

- **TP**: Pixels corretamente classificados.
- **FP**: Pixels incorretamente classificados.
- **FN**: Pixels que foram ignorados

![[Pasted image 20241122125209.png|center]]

Cálculo da **Mean IoU** utilizando a função nativa do Keras:
```python
#avaliando pela metrica "intersection over union"
from keras.metrics import MeanIoU
n_classes = 9
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
```
  **Mean IoU** (IoU médio) é calculada como a média dos IoUs de todas as classes presentes:

### 5.2.  Testes e Avaliação
Nesta seção são descritos os tipos de testes realizados para avaliar o funcionamento da implementação e a capacidade de generalização do modelo. A avaliação foi realizada em duas etapas principais:  testes de implementação e  generalização com o dataset sintético, e testes de generalização com o dataset de imagens reais. Os primeiros testes foram feitos durante a implementação, com o conjunto de validação, para verificar se o modelo e o pipeline de processamento estavam funcionando corretamente, seguidos de um novo  teste utilizando um conjunto nunca visto pelo modelo. Em seguida, novos modelos foram treinados com o dataset de imagens reais, que por ser muito escasso, demandou mudanças significativas nas estratégias de treinamento e arquitetura da rede.

### ==Dataset Sintético==
#### 5.2.1. Teste1: 

**Resultados**
![[image.png]]
No primeiro teste eu não fazia ideia de onde estava o erro. A materialização dos conjuntos revelou problemas no pipeline do pré processamento. Os principais erros identificados foram:

-  As imagens de ultrassom foram convertidas da escala de cinza para os índices de classe, quando não deveriam.
- O modelo foi compilado com *categorical_cross_entropy*, mas máscaras não estavam no formato esperado, como matrizes de vetores one-hot.

Esses erros comprometeram o funcionamento do modelo mas deram pistas para o ajuste do processo.

#### 5.2.2. Teste 2

**Resultados:** 
![[teste2 1.jpeg|center]]
![[teste2a 2.jpg|center]]
![[pred1.png|center|]]

Após corrigir os erros identificados no primeiro teste, os resultados foram medíocres mas forneceram informações importantes:

- A U-net implementada passou a treinar corretamente e realizar predições.
- Apenas o conjunto de teste foi normalizado, causando inconsistências entre os dados de treino e teste.
- O dataset está desbalanceado, com algumas classes tendo mais representatividade que outras, como o fígado, por exemplo.

Apesar disso, o modelo foi capaz de aprender algumas características gerais das imagens ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧.

#### 5.2.3.  Teste 3

- 832 Imagens de treino;
- 92 imagens de validação;

##### **Resultados:**
1![[pred2-11.png]]
2![[pred2-5.png]]3![[pred2-4.png]]
4![[pred2-13.png]]
5![[pred2-1.png]]
Após as correções, o modelo foi capaz de aprender as características das imagens e conseguiu gerar máscaras muito próximas das originais, classificando corretamente vários órgãos. Algumas deficiências e dificuldades foram identificadas:

- O modelo confundiu algumas classes e gerou máscaras de cores trocadas.
- Mesmo classificando corretamente, o formato dos órgãos está distorcido em algumas predições.
- As classes menos representadas foram as mais afetadas pelos erros de classificação e distorção, destacando o desbalanceamento do dataset.

Apesar disso, o modelo executou corretamente a tarefa proposta e seu funcionamento foi validado. ദ്ദി ˉ͈̀꒳ˉ͈́ )✧
##### **Avaliação:** 

**Perda de treino e validação:**
![[Pasted image 20241121172301.png|center|350]]

- A perda de treinamento decresce continuamente,  indicando que o modelo ajustou os pesos e aprendeu com os dados.
- A oscilação e valor da perda de validação foram maiores que os do treinamento, indicando ***overfitting*** e talvez problemas com ruídos nas imagens.

**Acurácia de treino e validação:**
![[Pasted image 20241121172313.png|center|350]]

- A acurácia de treinamento foi de 98% e com crescimento constante, indicando que o modelo conseguiu identificar os padrões nos dados de treinamento.
- Acurácia de validação oscilou muito e seu valor foi menor que a de treinamento, indicando ***overfitting***.

**Intersection over Union:**

`Mean IoU = 0.33217388`

**Métricas de treinamento:**
`- accuracy: 0.9306` 
`- loss: 0.3087` 
`- Accuracy is = 93.20508241653442 %`

#### 5.2.4. Teste de generalização
Utilizando um conjunto de teste com imagens não usadas no treinamento.
- 708 Imagens de treino;
- 125 Imagens validação;
- 93 Imagens de teste;
##### Resultados:
Uma seleção de resultados bons e ruins
1![[plot_35.png]]
2![[plot_14.png]]
3![[plot_47.png]]
4![[plot_27.png]]
5![[plot_72.png]]
6![[plot_66.png]]

- **Predições 1 e 2:** Resultados satisfatórios. O modelo foi capaz de gerar máscaras próximas as reais, com a localização e classificação corretas.
- **Predições 3 e 4:** Resultados medianos. O modelo gerou boas máscaras para algumas classes, e outras foram ignoradas. Também houveram erros de classificação.
- **Predições 5 e 6:** Resultados ruins. O modelo confundiu e ignorou algumas classes, e gerou máscaras inexistentes.

Mesmo nos melhores resultados, as máscaras geradas tem bordas distorcidas em relação ao ground thuth. Os piores resultados podem ter sido influenciados pela natureza ruidosa das imagens, e pelo dataset desbalanceado.
##### Avaliação: 

**Perda de treino e validação:**
![[Pasted image 20241121215646.png|center|350]]
- A perda de treino é baixa durante todo o treinamento, um forte indício de overfitting.
- A queda da perda no treino de validação indica que o modelo está aprendendo bem no início.

**Acurácia de treino e validação:**
![[Pasted image 20241121215655.png|center|350]]
- A diferença entre a acurácia de treino e validação reforça o overfitting.

O treinamento deste teste, embora com a mesma acurácia, foi melhor que o anterior. O aumento do conjunto de validação certamente influenciou nisso, pois foi o único parâmetro alterado.

**Métricas de treinamento:**
`-accuracy: 0.9322` 
`-loss: 0.2672 
`-Accuracy is = 93.42482089996338 %`

**Intersection over Union:**
`Mean IoU val = 0.3306403`
`Mean IoU TESTE = 0.3960948`

Apesar de a acurácia estar alta, ela *não é a **métrica ideal** para problemas de segmentação, por ser influenciada pelo desbalanceamento  dos dados. Como o modelo infere para cada pixel da imagem, ele acaba classificando corretamente as classes majoritárias, como o fundo preto que é a classe mais bem representada no dataset.

A métrica correta - **IoU(Intersection over Union)** - está longe do ideal, apenas 39% da área predita pelo  modelo coincide corretamente com o *ground truth*. Os 61% restantes correspondem a falsos positivos (áreas preditas que não existem no ground truth) e falsos negativos (áreas do ground truth que não foram preditas).

### Dataset Real

#### Teste 1: igual aos testes sinteticos

### Funções de perda customizadas

#### K-Fold Cross validation
### 5.3. Melhorias propostas

Para melhorar os resultados do modelo, as seguintes implementações podem ser feitas:

1. **Data Augmentation:** O trabalho ''Automated Deep Learning-Based Finger Joint Segmentation in 3-D Ultrasound Images With Limited Dataset'''  sugere os seguintes parâmetros de data augmentation para datasets de ultrassom;

	**Transformações geométricas aleatórias:**
    
    - Translação (horizontal e vertical) dentro de [-30%, 30%].
    - Rotação dentro de [-15°, 15°].
    - **Cisalhamento (Shear) dentro de [-5°, 5°]**.
   
	**Variações aleatórias de brilho e contraste:**
    
    - **Brilho** e **Contraste** ajustados aleatoriamente para simular diferentes configurações de equipamentos de ultrassom.
    
![[Pasted image 20250315172552.png]]
![[Pasted image 20250315172616.png]]
![[Pasted image 20250315172629.png]]
![[Pasted image 20250315172709.png]]

2. **Balanceamento dos dados:** Aplicar pesos de classe na função para dar mais importância às classes menos representadas.
   
3. **Testar  diferentes funções de perda:
	- **Jaccard Coefficient Loss:** Otimiza os pesos utilizando a métrica Intersection over Union. Útil para datasets desbalanceados.
	- **Dice Coeficient Loss:**

4. **Testar otimizadores diferentes como AdamW**

5. **Diferentes visualizações de resultados  e métricas:**
   - Visualização da intersecção entre a máscara gerada e o ground truth, e imagem original;
   - Visualização geral das predições com uma matriz de confusão utilizando um *threshhold* para o valor de IoU;
   - Grafico da curva AUC-ROC;
   - Geração de legendas na predição de acordo com as classes presentes na imagem;
## 6. Trabalhos futuros

Neste projeto foi desenvolvida e treinada uma arquitetura U-Net para segmentação semântica multiclasse em imagens de ultrassonografia, simuladas a partir de dados volumétricos de tomografia computadorizada. A evolução do projeto seria, naturalmente, sua adaptação para processar as imagens de reais de ultrassom do dataset original. Isso pode ser alcançado utilizando a técnica de **transfer learning** aproveitando os pesos treinados na etapa atual, juntamente com  data augmentation para contornar o problema da pequena amostra de imagens anotadas. Diferentes variações da U-net também podem ser implementadas, como **Attention U-net** que utiliza mecanismos de atenção para focar em regiões importantes da imagem, ou a utilização de funões de perda customizadas..

Outra possibilidade seria a adaptação da rede para processar dados volumétricos e realizar **segmentação semântica 3D**. O pipeline desenvolvido no projeto é flexível e permite o processamento de diferentes conjuntos de dados para resolver problemas variados de segmentação.

