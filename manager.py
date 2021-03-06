import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn
from urllib.parse import urlparse

seaborn.set()
np.set_printoptions(threshold=sys.maxsize)

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape

import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts


batch_size = 500
latent_dim = 8
dropout_rate = 0.3
start_lr = 0.001
num_classes = 41


if __name__ == "__main__":
    print("start")
    # загружаем изображения
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory="georgian_letters/HRS_Training_data/data",
        validation_split = 0.2,
        subset="training",
        seed=123,
        image_size=(28,28),
        batch_size=32)

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory="georgian_letters/HRS_Training_data/data",
        validation_split = 0.2,
        subset="validation",
        seed=123,
        image_size=(28,28),
        batch_size=32)

    # проверяем названия классов
    class_names = train_dataset.class_names
    print(class_names)

    # Отображаем на графике грузинский шрифт
    plt.figure(figsize=(3, 3))
    plt.rcParams.update({'font.family' : 'Sylfaen'})

    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()

    # Проверяем размеры в батчах
    for image_batch, labels_batch in train_dataset:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # Конфигурируем роизводительность датасета
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    #train_dataset = train_dataset.map(lambda x, y: (tf.divide(x, 255), y))
    #y = np.concatenate([y for x, y in train_dataset], axis=0)

    for images, labels in train_dataset.take(-1):  # only take first element of dataset
        numpy_train_images = images.numpy()
        numpy_train_labels = labels.numpy()

    for images, labels in val_dataset.take(-1):  # only take first element of dataset
        numpy_val_images = images.numpy()
        numpy_val_labels = labels.numpy()

    numpy_train_images = numpy_train_images / 255.
    numpy_val_images = numpy_val_images / 255.

    # Размерность кодированного представления
    encoding_dim = 49*2*3

    # Энкодер
    # Входной плейсхолдер
    input_img = Input(shape=(28, 28, 3))
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    x = Dense(encoding_dim*3, activation='relu')(flat_img)
    x = Dense(encoding_dim*2, activation='relu')(x)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim, activation="linear")(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim, ))
    x = Dense(encoding_dim*2, activation='relu')(input_encoded)
    x = Dense(encoding_dim*3, activation='relu')(x)
    flat_decoded = Dense(28*28*3, activation="sigmoid")(input_encoded)
    decoded = Reshape((28, 28, 3))(flat_decoded)

    d_encoder = Model(input_img, encoded, name="encoder")
    d_decoder = Model(input_encoded, decoded, name = "decoder")
    d_autoencoder = Model(input_img, d_decoder(d_encoder(input_img)), name = "autoencoder")

    d_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    d_autoencoder.summary()

    with mlflow.start_run():
        epochs = 800
        batch_size = 256

        history = d_autoencoder.fit(numpy_train_images, numpy_train_images, 
                        epochs=epochs, batch_size=batch_size, shuffle=True, 
                        validation_data=(numpy_val_images, numpy_val_images))

        log_param("epochs", epochs)
        log_param("batch_size", batch_size)
        log_param("encoding_dim", encoding_dim)
        log_metric("loss", history.history["loss"][-1])
        log_metric("val_loss", history.history["val_loss"][-1])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)

        #if tracking_url_type_store != "file":
        #    mlflow.sklearn.log_model(autoencoder, "model", registered_model_name="autoencoder")
        #else:
        #    mlflow.sklearn.log_model(autoencoder, "model")


"""
# Отрисовка букв
def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    
    plt.figure(figsize=(2*n, 2*len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i*n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

n = 10
imgs = numpy_val_images[:n]
encoded_imgs = d_encoder.predict(imgs, batch_size=batch_size)
encoded_imgs[0]

decoded_imgs = d_decoder.predict(encoded_imgs, batch_size=batch_size)
#decoded_imgs = decoded_imgs * 255.
plot_digits(imgs, decoded_imgs)
"""