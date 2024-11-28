import tensorflow as tf
from tensorflow.keras import Model

# Bloco convolucional utilizado em todas as etapas
def ConvBlock(tensor, num_feature):

  x = tf.keras.layers.Conv2D(num_feature, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(tensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Conv2D(num_feature, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  return x

def ponte(tensor, num_feature):
  x = ConvBlock(tensor,num_feature)
  return x

# Seção encoder
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

"""### estrutura da u-net"""

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