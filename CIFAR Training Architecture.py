from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, GaussianNoise
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.datasets import cifar100
import matplotlib.pyplot as plt
from keras_radam import RAdam
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def createModel(args, opt):
    numNodes = [9, 18, 36, 72, 144, 288, 576]
    k = (args.kernel_size * 2) + 1

    model = Sequential()
    model.add(GaussianNoise(args.gauss_noise, input_shape=(32,32,3)))
    model.add(Conv2D(filters=numNodes[0], kernel_size=(k, k), padding='same'))      # Convolution 1
    model.add(Activation('relu'))                                                   #   ReLU 1
    model.add(Conv2D(filters=numNodes[1], kernel_size=(k, k), padding='same'))      # Convlution 2
    model.add(Activation('relu'))                                                   #   ReLU 2
    model.add(Conv2D(filters=numNodes[2], kernel_size=(k, k), padding='same'))      # Convolution 3
    model.add(Activation('relu'))                                                   #   ReLU 3
    model.add(Dropout(args.dropout_conv))                                           #       Dropout 1
    model.add(MaxPooling2D(pool_size=2))                                            #       Max Pooling 1
    model.add(Conv2D(filters=numNodes[3], kernel_size=(k, k), padding='same'))      # Convolution 4
    model.add(Activation('relu'))                                                   #   ReLU 4
    model.add(Conv2D(filters=numNodes[4], kernel_size=(k, k), padding='same'))      # Convolution 5
    model.add(Activation('relu'))                                                   #   ReLU 5
    model.add(Conv2D(filters=numNodes[5], kernel_size=(k, k), padding='same'))      # Convolution 6
    model.add(Activation('relu'))                                                   #   ReLU 6
    model.add(Dropout(args.dropout_conv))                                           #       Dropout 2
    model.add(MaxPooling2D(pool_size=2))                                            #       Max Pooling 2
    model.add(Conv2D(filters=numNodes[6], kernel_size=(k, k), padding='same'))      # Convolution 7
    model.add(Activation('relu'))                                                   #   ReLU 7
    model.add(Flatten())                                                            #       Flatten
    model.add(Dense(numNodes[-1]))                                                  #       Dense 1 (FC)
    model.add(Activation('relu'))                                                   #       ReLU 8
    model.add(Dropout(args.dropout_dense))                                          #       Dropout 3
    model.add(Dense(100, activation='softmax'))                                     #       Dense 2 (FC)

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def labelMap():
    Map = np.zeros(100).astype('int')
    Map[[4, 30, 55, 72, 95]] = 0;       Map[[23, 33, 49, 60, 71]] = 10
    Map[[1, 32, 67, 73, 91]] = 1;       Map[[15, 19, 21, 31, 38]] = 11
    Map[[54, 62, 70, 82, 92]] = 2;      Map[[34, 63, 64, 66, 75]] = 12
    Map[[9, 10, 16, 28, 61]] = 3;       Map[[26, 45, 78, 79, 99]] = 13
    Map[[0, 51, 53, 57, 83]] = 4;       Map[[2, 11, 35, 46, 98]] = 14
    Map[[22, 39, 40, 86, 87]] = 5;      Map[[27, 29, 44, 78, 93]] = 15
    Map[[5, 20, 25, 84, 94]] = 6;       Map[[36, 50, 65, 74, 80]] = 16
    Map[[6, 7, 14, 18, 24]] = 7;        Map[[47, 52, 56, 59, 96]] = 17
    Map[[3, 42, 43, 88, 97]] = 8;       Map[[8, 13, 48, 58, 90]] = 18
    Map[[12, 17, 37, 68, 76]] = 9;      Map[[41, 69, 81, 85, 89]] = 19

    return Map

    # 0 aquatic_mammals                     4, 30, 55, 72, 95
    # 1 fish                                1, 32, 67, 73, 91
    # 2 flowers                             54, 62, 70, 82, 92
    # 3 food_containers                     9, 10, 16, 28, 61
    # 4 fruit_and_vegetables                0, 51, 53, 57, 83
    # 5 household_electrical_devices        22, 39, 40, 86, 87
    # 6 household_furniture                 5, 20, 25, 84, 94
    # 7 insects                             6, 7, 14, 18, 24
    # 8 large_carnivores                    3, 42, 43, 88, 97
    # 9 large_man-made_outdoor_things       12, 17, 37, 68, 76
    # 10 large_natural_outdoor_scenes       23, 33, 49, 60, 71
    # 11 large_omnivores_and_herbivores     15, 19, 21, 31, 38
    # 12 medium_mammals                     34, 63, 64, 66, 75
    # 13 non-insect_invertebrates           26, 45, 78, 79, 99
    # 14 people                             2, 11, 35, 46, 98
    # 15 reptiles                           27, 29, 44, 78, 93
    # 16 small_mammals                      36, 50, 65, 74, 80
    # 17 trees                              47, 52, 56, 59, 96
    # 18 vehicles_1                         8, 13, 48, 58, 90
    # 19 vehicles_2                         41, 69, 81, 85, 89

def train(args):
    ## Prepare data
    print("Loading data..." + '\n')
    (trainX, trainF), (testX, testF) = cifar100.load_data(label_mode='fine')
    (_,      trainC), (_,     testC) = cifar100.load_data(label_mode='coarse')

    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    valX = trainX[40000:, :, :, :]              # (10000,32,32,3)           Validation Data
    trainX = trainX[:40000, :, :, :]            # (40000,32,32,3)           Training Data
    valF = to_categorical(trainF[40000:])       # (10000,100)               Validation Fine Labels
    trainF = to_categorical(trainF[:40000])     # (40000,100)               Training Fine Labels
    valC = trainC[40000:]                       # (10000,)                  Validation Coarse Labels
    trainC = trainC[:40000]                     # (40000,)                  Training Coarse Labels

    ## Extract useful variables
    batchSize = args.batch_size
    trainPath = args.save_dir
    if trainPath[-1] != '/':
        trainPath += '/'


    optimizers = {"RAdam": RAdam(min_lr=args.learning_rate), "Adam": Adam(lr=args.learning_rate), "SGD": SGD(lr=args.learning_rate)}
    for optimizer, opt_object in optimizers.items():
        modelPath = trainPath + "bestModel{}.hdf5".format(optimizer)
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True)
        earlyStop = EarlyStopping(monitor='val_acc', patience=10)

        model = createModel(args, opt_object)
        H = model.fit(x=trainX, y=trainF, validation_data=(valX, valF), batch_size=batchSize, epochs=args.epochs,
                              verbose=1, callbacks=[checkpointer, earlyStop], shuffle=True)
        print("{} hist size: {}".format(optimizer, H.shape))
        model.load_weights(modelPath)
        predictions = model.predict(testX, batch_size=batchSize)
        getAccuracy(predictions, testF, testC)

    ############ Visualizing training history #################
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="Train Loss")
    plt.plot(H.history["val_loss"], label="Val Loss")
    plt.plot(H.history["acc"], label="Train Accuracy")
    plt.plot(H.history["val_acc"], label="Val Accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(trainPath + "accuracyAndLoss.jpg")

    print("Predicting off of best epoch....")
    plt.show()
    ###########################################################

def getAccuracy(preds, testF, testC):
    totalCount = len(preds)
    fineCount, coarseCount = 0, 0
    map = labelMap()
    for idx, p in enumerate(preds):
        print(p)
        if p == np.argmax(testF[idx,:]):
            fineCount += 1
        if map[p] == testC[idx]:
            coarseCount += 1

    fineAcc = fineCount / totalCount
    coarseAcc = coarseCount / totalCount
    print("Accuracy on Fine Labels: {}%".format(np.round(fineAcc * 100, decimals=2)))
    print("Accuracy on Coarse Labels: {}%".format(np.round(coarseAcc * 100, decimals=2)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs',        type=int,    default=3,                  help='Max number of epochs')
    parser.add_argument('-b', '--batch-size',    type=int,    default=64,                  help='Number of images per batch.')
    parser.add_argument('-l', '--learning-rate', type=float,  default=0.00001,             help='Learning rate for RAdam optimizer.')
    parser.add_argument('-c', '--dropout-conv',  type=float,  default=0.1,                 help='Dropout rate applied after Conv layers. Range: 0-0.15')
    parser.add_argument('-d', '--dropout-dense', type=float,  default=0.15,                help='Dropout rate applied after Dense layer. Range: 0.1-0.3')
    parser.add_argument('-g', '--gauss-noise',   type=float,  default=0.01,                help='Amount of gaussian noise applied to input image.')
    parser.add_argument('-k', '--kernel-size',   type=int,    default=1,                   help='Actual Kernel Size = (kernel_size * 2) + 1. Range: 1-3')
    parser.add_argument('-m', '--model-name',    type=str,    default="bestModel100.hdf5", help='Name of model file')
    parser.add_argument('-p', '--save-dir',      type=str,    default="/home/ubuntu//",
                        help='Directory to save model file and history plot')

    (args, _) = parser.parse_known_args()
    train(args)
