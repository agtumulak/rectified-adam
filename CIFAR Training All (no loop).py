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
    print("Model has {} paramters".format(model.count_params()))
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
    trainF = to_categorical(trainF)                         # (40000,100)               Training Fine Labels
    testF_cat = to_categorical(testF)                       # (10000,100)               Test Fine Labels

    ## Extract useful variables
    permu = to_categorical(np.array([4, 30, 55, 72, 95,   1, 32, 67, 73, 91,   54, 62, 70, 82, 92,  9, 10, 16, 28, 61,
                                     0, 51, 53, 57, 83,   22, 39, 40, 86, 87,  5, 20, 25, 84, 94,   6, 7, 14, 18, 24,
                                     3, 42, 43, 88, 97,   12, 17, 37, 68, 76,  23, 33, 49, 60, 71,  15, 19, 21, 31, 38,
                                     34, 63, 64, 66, 75,  26, 45, 78, 79, 99,  2, 11, 35, 46, 98,   27, 29, 44, 78, 93,
                                     36, 50, 65, 74, 80,  47, 52, 56, 59, 96,  8, 13, 48, 58, 90,   41, 69, 81, 85, 89]))
    batchSize = args.batch_size
    trainPath = args.save_dir
    if trainPath[-1] != '/':
        trainPath += '/'

    train_loss = ([0] * args.epochs)
    train_acc = np.copy(train_loss)
    val_loss = np.copy(train_loss)
    val_acc = np.copy(train_loss)

    ## Actual training happens here
    Accuracies = np.zeros((3, 3))
    RAdam_opt = RAdam(min_lr=args.learning_rate)
    Adam_opt = Adam(lr=args.learning_rate / 10)
    SGD_opt = SGD(lr=args.learning_rate)

    modelPath = trainPath + "bestModelAdam_RN.hdf5"
    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.6, min_lr=0.000005)
    Adam_model = createModel(args, Adam_opt)
    Adam_H = Adam_model.fit(x=trainX, y=trainF, validation_data=(testX, testF_cat), batch_size=batchSize,
                            epochs=args.epochs, verbose=1, callbacks=[checkpointer, lr_reduction], shuffle=True)
    train_loss = np.vstack((train_loss, Adam_H.history["loss"]))
    train_acc = np.vstack((train_acc, Adam_H.history["acc"]))
    val_loss = np.vstack((val_loss, Adam_H.history["val_loss"]))
    val_acc = np.vstack((val_acc, Adam_H.history["val_acc"]))
    Adam_model.load_weights(modelPath)
    Adam_pred = Adam_model.predict(testX, batch_size=batchSize)  # (10000,100)
    Accuracies[1, :] = getAccuracy(Adam_pred, testF, testC, "Adam")
    Adam_conf = permu @ confusion_matrix(testF, np.argmax(Adam_pred, axis=1)) @ permu.T
    del modelPath, checkpointer, lr_reduction, Adam_model, Adam_H, Adam_pred


    modelPath = trainPath + "bestModelRAdam_RN.hdf5"
    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.6, min_lr=0.000005)
    RAdam_model = createModel(args, RAdam_opt)
    RAdam_H = RAdam_model.fit(x=trainX, y=trainF, validation_data=(testX, testF_cat), batch_size=batchSize, epochs=args.epochs,
                          verbose=1, callbacks=[checkpointer, lr_reduction], shuffle=True)
    train_loss = np.vstack((train_loss, RAdam_H.history["loss"]))
    train_acc = np.vstack((train_acc, RAdam_H.history["acc"]))
    val_loss = np.vstack((val_loss, RAdam_H.history["val_loss"]))
    val_acc = np.vstack((val_acc, RAdam_H.history["val_acc"]))
    RAdam_model.load_weights(modelPath)
    RAdam_pred = RAdam_model.predict(testX, batch_size=batchSize)                    # (10000,100)
    Accuracies[0,:] = getAccuracy(RAdam_pred, testF, testC, "RAdam")
    RAdam_conf = permu @ confusion_matrix(testF, np.argmax(RAdam_pred, axis=1)) @ permu.T
    del modelPath, checkpointer, lr_reduction, RAdam_model, RAdam_H, RAdam_pred


    modelPath = trainPath + "bestModelSGD_RN.hdf5"
    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.6, min_lr=0.000005)
    SGD_model = createModel(args, SGD_opt)
    SGD_H = SGD_model.fit(x=trainX, y=trainF, validation_data=(testX, testF_cat), batch_size=batchSize,
                             epochs=args.epochs, verbose=1, callbacks=[checkpointer, lr_reduction], shuffle=True)
    train_loss = np.vstack((train_loss, SGD_H.history["loss"]))
    train_acc = np.vstack((train_acc, SGD_H.history["acc"]))
    val_loss = np.vstack((val_loss, SGD_H.history["val_loss"]))
    val_acc = np.vstack((val_acc, SGD_H.history["val_acc"]))
    SGD_model.load_weights(modelPath)
    SGD_pred = SGD_model.predict(testX, batch_size=batchSize)                       # (10000,100)
    Accuracies[2, :] = getAccuracy(SGD_pred, testF, testC, "SGD")
    SGD_conf = permu @ confusion_matrix(testF, np.argmax(SGD_pred, axis=1)) @ permu.T
    del modelPath, checkpointer, lr_reduction, SGD_model, SGD_H, SGD_pred

    print(Accuracies)

    ########################### Visualizing training history ##########################
    train_loss = np.delete(train_loss, (0), axis=0).T
    train_acc = np.delete(train_acc, (0), axis=0).T
    val_loss = np.delete(val_loss, (0), axis=0).T
    val_acc = np.delete(val_acc, (0), axis=0).T
    print('Training Loss:\n', train_loss)
    print('Validation Loss:\n', val_loss)
    print('Training Accuracy:\n', train_acc)
    print('Validation Accuracy:\n', val_acc)

    plt.style.use("ggplot")
    #plt.plot(train_acc[:,0], '-r')
    plt.plot(val_acc[:, 0], '-b')
    #plt.plot(train_acc[:,1], '-g')
    plt.plot(val_acc[:, 1], '-k')
    #plt.plot(train_acc[:,2], '-m')
    plt.plot(val_acc[:, 2], '-', color='#f4d100')
    plt.xlim((1, args.epochs))
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend(("Adam Accuracy", "RAdam Accuracy", "SGD Accuracy"), loc="lower right")
    #plt.legend(("RAdam Train", "RAdam Validation", "Adam Train", "Adam Validation","SGD Train", "SGD Validation"), loc="best")
    plt.savefig(trainPath + "RAdamAdamSGD_accuracy_RN.jpg")

    plt.figure(2)
    #plt.plot(train_loss[:, 0], '-r')
    plt.plot(val_loss[:, 0], '-b')
    #plt.plot(train_loss[:, 1], '-g')
    plt.plot(val_loss[:, 1], '-k')
    #plt.plot(train_loss[:, 2], '-m')
    plt.plot(val_loss[:, 2], '-', color='#f4d100')
    plt.xlim((1, args.epochs))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(("Adam Loss", "RAdam Loss", "SGD Loss"), loc="upper right")
    #plt.legend(("RAdam Train", "RAdam Validation", "Adam Train", "Adam Validation", "SGD Train", "SGD Validation"), loc="upper right")
    plt.savefig(trainPath + "RAdamAdamSGD_loss_RN.jpg")

    plt.figure(3)
    labels = np.array(["RAdam", "Adam", "SGD"])
    x = np.arange(3)
    width = 0.3
    rects1 = plt.bar(x - width, Accuracies[:, 0], width, color='#fc7c00')
    rects2 = plt.bar(x, Accuracies[:, 1], width, color='g')
    rects3 = plt.bar(x + width, Accuracies[:, 2], width, color='#bd78f9')
    bars = [rects1, rects2, rects3]

    plt.xticks(x, labels)
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.legend(("Fine Labels", "Fine Labels (Top 3)", "Coarse Labels"), loc="lower right")
    plt.title("RAdam/Adam/SGD Best Accuracy Metrics")

    for rects in bars:
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.savefig(trainPath + "RAdamAdamSGD_metrics_RN.jpg")

    plt.figure(4)
    sns.heatmap(RAdam_conf, linewidths=0, square=True, cmap='RdYlBu')
    plt.savefig(trainPath + "RAdam Confusion Permu.jpg")
    plt.figure(5)
    sns.heatmap(Adam_conf, linewidths=0, square=True, cmap='RdYlBu')
    plt.savefig(trainPath + "Adam Confusion Permu.jpg")
    plt.figure(6)
    sns.heatmap(SGD_conf, linewidths=0, square=True, cmap='')
    plt.savefig(trainPath + "SGD Confusion Permu.jpg")

    plt.show()
    ##################################################################################

def getAccuracy(preds, testF, testC, optName):
    totalCount = len(preds)
    fineCount, coarseCount, top3Count = 0, 0, 0
    map = labelMap()
    for idx, p in enumerate(preds):
        #print(np.argmax(p), testF[idx,:][0], np.argsort(p)[-3:][::-1])
        if np.argmax(p) == testF[idx,:][0]:
            fineCount += 1
        if testF[idx, :][0] in np.argsort(p)[-3:]:
            top3Count += 1
        if map[np.argmax(p)] == testC[idx]:
            coarseCount += 1

    fineAcc = np.round((fineCount / totalCount) * 100, decimals=1)
    top3Acc = np.round((top3Count / totalCount) * 100, decimals=1)
    coarseAcc = np.round((coarseCount / totalCount) * 100, decimals=1)
    print(" ----- {} accuracy on Fine Labels: {}%".format(optName, fineAcc))
    print(" ----- {} Top 3 accuracy on Fine Labels: {}%".format(optName, top3Acc))
    print(" ----- {} accuracy on Coarse Labels: {}%".format(optName, coarseAcc))
    print('\n')
    return (fineAcc, top3Acc, coarseAcc)

if __name__ == "__main__":
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
    train(args)
