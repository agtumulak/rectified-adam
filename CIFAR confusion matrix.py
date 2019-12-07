from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, GaussianNoise
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.initializers import Zeros, Ones, Constant, RandomNormal
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.datasets import cifar100
import matplotlib.pyplot as plt
from keras_radam import RAdam
import tensorflow as tf
import seaborn as sns
import numpy as np
import argparse
import logging, os
logging.disable(logging.WARNING)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(0)
tf.set_random_seed(2)

def createModel(args, opt):
    numNodes = [8, 16, 32, 64, 128, 256, 512]
    k = (args.kernel_size * 2) + 1
    w8s = {"Zeros": Zeros(), "Ones": Ones(), "Constant": Constant(value=0.2),
           "RandNormal": RandomNormal(mean=0.0, stddev=0.05, seed=0)}

    model = Sequential()
    model.add(GaussianNoise(args.gauss_noise, input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=numNodes[0], kernel_size=(k, k), padding='same',
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convolution 1
    model.add(Activation('relu'))                                                                   #   ReLU 1
    model.add(Conv2D(filters=numNodes[1], kernel_size=(k, k), padding='same',                       #
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convlution 2
    model.add(Activation('relu'))                                                                   #   ReLU 2
    model.add(Dropout(args.dropout_conv))                                                           #       Dropout 1
    model.add(MaxPooling2D(pool_size=2))                                                            #       Max Pooling 1
                                                                                                    #
    model.add(Conv2D(filters=numNodes[2], kernel_size=(k, k), padding='same',                       #
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convolution 3
    model.add(Activation('relu'))                                                                   #   ReLU 3
    model.add(Conv2D(filters=numNodes[3], kernel_size=(k, k), padding='same',                       #
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convolution 4
    model.add(Activation('relu'))                                                                   #   ReLU 4
    model.add(Dropout(args.dropout_conv))                                                           #       Dropout 2
    model.add(MaxPooling2D(pool_size=2))                                                            #       Max Pooling 2
                                                                                                    #
    model.add(Conv2D(filters=numNodes[4], kernel_size=(k, k), padding='same',                       #
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convolution 5
    model.add(Activation('relu'))                                                                   #   ReLU 5
    model.add(Conv2D(filters=numNodes[5], kernel_size=(k, k), padding='same',                       #
                     kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))             # Convolution 6
    model.add(Activation('relu'))                                                                   #   ReLU 6
    model.add(Dropout(args.dropout_conv))                                                           #       Dropout 3
    model.add(MaxPooling2D(pool_size=2))                                                            #       Max Pooling 3
                                                                                                    #
    model.add(Flatten())                                                                            #       Flatten
    model.add(Dense(numNodes[6], kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s])) #       Dense 1 (FC)
    model.add(Activation('relu'))                                                                   #       ReLU 8
    model.add(Dropout(args.dropout_dense))                                                          #       Dropout 4
    model.add(Dense(100, activation='softmax',                                                      #
                    kernel_initializer=w8s[args.w8s], bias_initializer=w8s[args.w8s]))              #       Dense 2 (FC)
                                                                                                    #
    #print(model.summary())
    #print("Model has {} paramters".format(model.count_params()))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs',        type=int,    default=2,              help='Max number of epochs')
parser.add_argument('-b', '--batch-size',    type=int,    default=64,              help='Number of images per batch.')
parser.add_argument('-l', '--learning-rate', type=float,  default=0.01,            help='Learning rate for RAdam optimizer.')
parser.add_argument('-c', '--dropout-conv',  type=float,  default=0.25,            help='Dropout rate applied after Conv layers. Range: 0-0.15')
parser.add_argument('-d', '--dropout-dense', type=float,  default=0.25,            help='Dropout rate applied after Dense layer. Range: 0.1-0.3')
parser.add_argument('-g', '--gauss-noise',   type=float,  default=0.1,             help='Amount of gaussian noise applied to input image.')
parser.add_argument('-k', '--kernel-size',   type=int,    default=2,               help='Actual Kernel Size = (kernel_size * 2) + 1. Range: 1-3')
parser.add_argument('-w', '--w8s',           type=str,    default="RandNormal",    help='Defines way weights will be initialized')
parser.add_argument('-p', '--save-dir',      type=str,    default="/home/ubuntu/", help='Directory to save model file and history plot')

(args, _) = parser.parse_known_args()

modelPath = "/home/ubuntu/"
batchSize = 64

(trainX, trainF), (testX, testF) = cifar100.load_data(label_mode='fine')
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
trainF = to_categorical(trainF)  # (40000,100)               Training Fine Labels
testF_cat = to_categorical(testF)  # (10000,100)               Test Fine Labels
permu = to_categorical(np.array([4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 28, 61,
                                 0, 51, 53, 57, 83, 22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 6, 7, 14, 18, 24,
                                 3, 42, 43, 88, 97, 12, 17, 37, 68, 76, 23, 33, 49, 60, 71, 15, 19, 21, 31, 38,
                                 34, 63, 64, 66, 75, 26, 45, 78, 79, 99, 2, 11, 35, 46, 98, 27, 29, 44, 78, 93,
                                 36, 50, 65, 74, 80, 47, 52, 56, 59, 96, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89]))
permu20 = np.kron(np.eye(20), np.ones((5,1)))


RAdam_model = createModel(args, RAdam(min_lr=args.learning_rate))
RAdam_model.load_weights(modelPath + "bestModelRAdam_RN.hdf5")
Adam_model = createModel(args, Adam(lr=args.learning_rate / 10))
Adam_model.load_weights(modelPath + "bestModelAdam_RN.hdf5")
SGD_model = createModel(args, SGD(lr=args.learning_rate))
SGD_model.load_weights(modelPath + "bestModelSGD_RN.hdf5")


RAdam_pred = RAdam_model.predict(testX, batch_size=batchSize)
RAdam_conf100 = permu @ confusion_matrix(testF, np.argmax(RAdam_pred, axis=1)) @ permu.T
RAdam_conf20 = permu20.T @ RAdam_conf100 @ permu20

Adam_pred = Adam_model.predict(testX, batch_size=batchSize)
Adam_conf100 = permu @ confusion_matrix(testF, np.argmax(Adam_pred, axis=1)) @ permu.T
Adam_conf20 = permu20.T @ Adam_conf100 @ permu20

SGD_pred = SGD_model.predict(testX, batch_size=batchSize)
SGD_conf100 = permu @ confusion_matrix(testF, np.argmax(SGD_pred, axis=1)) @ permu.T
SGD_conf20 = permu20.T @ SGD_conf100 @ permu20

labels = ["Aqua. Mamm.", "Fish", "Flowers", "Containers", "Fruit & Veg.", "Elec. Devices",
              "Furniture", "Insects", "Carnivores", "Buildings", "Natural Scene", "Herbivores",
              "Mammals M.", "Invertebrates", "People", "Reptiles", "Mammals S", "Trees", "Vehicles 1", "Vehicles 2"]
x = np.arange(20)

plt.figure(1)
sns.heatmap(RAdam_conf100, linewidths=0, square=True, cmap='RdYlBu')
plt.savefig(modelPath + "RAdam BestConfusion Permu.jpg")
plt.figure(2)
sns.heatmap(Adam_conf100, linewidths=0, square=True, cmap='RdYlBu')
plt.savefig(modelPath + "Adam BestConfusion Permu.jpg")
plt.figure(3)
sns.heatmap(SGD_conf100, linewidths=0, square=True, cmap='RdYlBu')
plt.savefig(modelPath + "SGD BestConfusion Permu.jpg")
plt.figure(4)
plt.figure(figsize=((9,9)))
sns.heatmap(RAdam_conf20, linewidths=0, square=True, cmap='RdYlBu')
plt.xticks(x+0.5, labels, rotation='vertical'); plt.yticks(x+0.5, labels, rotation='horizontal'); plt.margins(0.5)
plt.savefig(modelPath + "RAdam BestConfusion Permu20.jpg")
plt.figure(5)
plt.figure(figsize=((9,9)))
sns.heatmap(Adam_conf20, linewidths=0, square=True, cmap='RdYlBu')
plt.xticks(x+0.5, labels, rotation='vertical'); plt.yticks(x+0.5, labels, rotation='horizontal'); plt.margins(0.5)
plt.savefig(modelPath + "Adam BestConfusion Permu20.jpg")
plt.figure(6)
plt.figure(figsize=((9,9)))
sns.heatmap(SGD_conf20, linewidths=0, square=True, cmap='RdYlBu')
plt.xticks(x+0.5, labels, rotation='vertical'); plt.yticks(x+0.5, labels, rotation='horizontal'); plt.margins(0.5)
plt.savefig(modelPath + "SGD BestConfusion Permu20.jpg")



