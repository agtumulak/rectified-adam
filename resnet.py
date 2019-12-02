#!/usr/bin/env python
import argparse
import tensorflow as tf
from os import path


kernel_size = 3


def get_resnet_block(n, superblock=None, block=None):
    # map sizes go down by factor; filters go up by factor
    factor = pow(2, n-1)
    filters = 16 * factor
    map_size = 32 // factor

    dim_change = superblock != 1 and block == 1
    if dim_change:
        inputs = tf.keras.Input(
                shape=(map_size * 2, map_size * 2, filters // 2))
        identity = tf.keras.layers.Conv2D(
                filters, 1, padding='same', strides=2)(inputs)
        x = tf.keras.layers.Conv2D(
                filters, kernel_size, padding='same', strides=2)(inputs)
    else:
        inputs = tf.keras.Input(shape=(map_size, map_size, filters))
        identity = inputs
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)

    # Convolution Path
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Sum them
    x = tf.keras.layers.add([identity, x])
    outputs = tf.keras.layers.Activation(tf.nn.relu)(x)
    name = 'ResNetBlock{superblock}{block}'.format(
            superblock='_{:d}'.format(superblock) if superblock else '',
            block='_{:d}'.format(block) if block else '')
    model = tf.keras.Model(inputs, outputs, name=name)
    model.summary()
    return model


def get_resnet():
    # Head
    inputs = tf.keras.Input(shape=(32, 32, 3), name='image')
    x = tf.keras.layers.Conv2D(16, kernel_size, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    # Body
    superblocks = [1, 2, 3]
    for superblock in superblocks:
        blocks = [1, 2, 3]
        for block in blocks:
            x = get_resnet_block(
                    superblock,
                    superblock=superblock,
                    block=block)(x)
    # Foot
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs, outputs, name="ResNet")
    model.summary()
    tf.keras.utils.plot_model(model, 'model.png', expand_nested=True)
    return model


def get_dataset():
    """
    He 2016:
    We follow the simple data augmentation in [24] for training: 4 pixels are
    padded on each side, and a 32×32 crop is randomly sampled from the padded
    image or its horizontal flip.
    """
    cifar10 = tf.keras.datasets.cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    # train_x, train_y = train_x[:5000], train_y[:5000]
    # train_x, test_x = train_x / 255., test_x / 255.

    # https://github.com/KaimingHe/deep-residual-networks/issues/5#issuecomment-183187647
    per_pixel_mean = train_x.mean(axis=0)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=4,
            height_shift_range=4,
            horizontal_flip=True,
            preprocessing_function=lambda image: image - per_pixel_mean)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda image: image - per_pixel_mean)
    train_datagen.fit(train_x)
    test_datagen.fit(train_x)
    train = train_datagen.flow(train_x, y=train_y, batch_size=128)
    test = test_datagen.flow(test_x, y=test_y)
    return train, test


def get_optimizer():
    return tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)  # He 4.2
    # return tf.keras.optimizers.Adam()


def get_scheduler():
    """
    He 2016:
    These models are trained with a mini-batch size of 128 on two GPUs. We
    start with a learning rate of 0.1, divide it by 10 at 32k and 48k
    iterations, and terminate training at 64k iterations
    """
    def schedule(epoch):
        if epoch >= 123:
            return 0.001
        elif epoch >= 82:
            return 0.01
        else:
            return 0.1
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)


def get_checkpoint():
    return tf.keras.callbacks.ModelCheckpoint(
            'weights.{epoch:03d}.hdf5', verbose=1)


if __name__ == '__main__':
    # Check for model file
    parser = argparse.ArgumentParser(description='An EECS 545 Project')
    parser.add_argument('--model-path', help='path to saved model')
    args = parser.parse_args()
    if args.model_path:
        model = tf.keras.models.load_model(args.model_path)
        initial_epoch = int(args.model_path.split('.')[1])
    else:
        model = get_resnet()
        model.compile(
                optimizer=get_optimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        initial_epoch = 0
    # Train
    train, test = get_dataset()
    model.fit_generator(
            train,
            initial_epoch=initial_epoch,
            epochs=164,
            validation_data=test,
            callbacks=[
                get_scheduler(),
                get_checkpoint(),
                tf.keras.callbacks.TensorBoard()])
