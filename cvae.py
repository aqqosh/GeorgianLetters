from keras.layers import Input, Dense 
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


def create_cvae():
    models = {}

    # Добавим Dropout и BatchNormalization
    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    # Энкодер
    input_img = Input(shape=(28, 28, 1))
    flatten_img = Flatten()(input_img)
    input_lbl = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([flatten_img, input_lbl])
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    # Предсказываем параметры распределений
    # Вместо того чтобы предсказывать стандартное отклонение, предсказываем логарифм вариации
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Сэмплирование из Q с трюком репараметризации
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"]  = Model([input_img, input_lbl], l, 'Encoder') 
    models["z_meaner"] = Model([input_img, input_lbl], z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model([input_img, input_lbl], z_log_var, 'Enc_z_log_var')

    # Декодер
    z = Input(shape=(latent_dim, ))
    input_lbl_d = Input(shape=(num_classes,), dtype='float32')
    x = concatenate([z, input_lbl_d])
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(28*28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    models["decoder"] = Model([z, input_lbl_d], decoded, name='Decoder')
    models["cvae"]    = Model([input_img, input_lbl, input_lbl_d], 
                                models["decoder"]([models["encoder"]([input_img, input_lbl]), input_lbl_d]), 
                                name="CVAE")
    models["style_t"] = Model([input_img, input_lbl, input_lbl_d], 
                                models["decoder"]([models["z_meaner"]([input_img, input_lbl]), input_lbl_d]), 
                                name="style_transfer")


    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, 28*28))
        decoded = K.reshape(decoded, shape=(batch_size, 28*28))
        xent_loss = 28*28*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/28/28

    return models, vae_loss

models, vae_loss = create_cvae()
cvae = models["cvae"]