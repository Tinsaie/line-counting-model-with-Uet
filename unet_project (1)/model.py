import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 720

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D(2)(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, skip])
    return conv_block(x, filters)

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b = conv_block(p4, 512)  # You can revert to 1024 if VRAM allows
    
    # Decoder
    d1 = decoder_block(b, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    
    return models.Model(inputs, outputs)

model = build_unet()
model.summary()


