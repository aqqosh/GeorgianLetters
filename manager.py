import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn

seaborn.set()
np.set_printoptions(threshold=sys.maxsize)

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

"""
list_ds = tf.data.Dataset.list_files(str("georgian_letters/HRS_Training_Data/data/"'*/*'))

def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]

  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [28, 28])

  return image, label

file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()

show(image, label)
"""

batch_size = 500
latent_dim = 8
dropout_rate = 0.3
start_lr = 0.001
num_classes = 41

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="georgian_letters/HRS_Training_data/data",
    validation_split = 0.2,
    subset="training",
    seed=123,
    #labels="inferred",
    #label_mode="categorical",
    image_size=(28,28),
    batch_size=32)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="georgian_letters/HRS_Training_data/data",
    validation_split = 0.2,
    subset="validation",
    seed=123,
    #labels="inferred",
    #label_mode="categorical",
    image_size=(28,28),
    batch_size=32)

class_names = train_dataset.class_names
print(class_names)

plt.figure(figsize=(3, 3))
plt.rcParams.update({'font.family' : 'Sylfaen'})

for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Размерность кодированного представления
encoding_dim = 49

# Энкодер
# Входной плейсхолдер
input_img = Input(shape=(28, 28, 1))
# Вспомогательный слой решейпинга
flat_img = Flatten()(input_img)
# Кодированное полносвязным слоем представление
encoded = Dense(encoding_dim, activation="relu")(flat_img)

# Декодер
# Раскодированное другим полносвязным слоем изображение
input_encoded = Input(shape=(encoding_dim, ))
flat_decoded = Dense(28*28, activation="sigmoid")(input_encoded)
decoded = Reshape((28, 28, 1))(flat_decoded)

encoder = Model(input_img, encoded, name="encoder")
decoder = Model(input_encoded, decoded, name = "decoder")
autoencoder = Model(input_img, decoder(encoder(input_img)), name = "autoencoder")

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()
autoencoder.fit(train_dataset, epochs=50, batch_size=256, shuffle=True, validation_data=val_dataset)

train_np = np.stack(list(train_dataset))
