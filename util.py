import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import cv2

def load_model():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("vertical"),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0.2)
    ])

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        data_augmentation,
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(32, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.load_weights('./checkpoint/percobaan2.weight')

    return model

def recognition(model, image):
    image = cv2.resize(image, (64,64), interpolation = cv2.INTER_AREA)
    R=np.zeros((64,64,3)).astype('uint8')
    R[:,:,0]=image
    R[:,:,1]=image
    R[:,:,2]=image
    R = cv2.flip(R,1)
    image = tf.convert_to_tensor(R)
    image = tf.expand_dims(image,axis=0)
    
    a = model.predict(image)

    return np.argmax(a), np.max(a)