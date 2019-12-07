#############################################
# View images in CIFAR-100 one-by-one       #
#   -Prints out coarse/fine labels for each #
#############################################

from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
import pickle
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def labelMap():
    Map = np.zeros(100).astype('int')
    Map[[4, 30, 55, 72, 95]] = 0;
    Map[[23, 33, 49, 60, 71]] = 10
    Map[[1, 32, 67, 73, 91]] = 1;
    Map[[15, 19, 21, 31, 38]] = 11
    Map[[54, 62, 70, 82, 92]] = 2;
    Map[[34, 63, 64, 66, 75]] = 12
    Map[[9, 10, 16, 28, 61]] = 3;
    Map[[26, 45, 78, 79, 99]] = 13
    Map[[0, 51, 53, 57, 83]] = 4;
    Map[[2, 11, 35, 46, 98]] = 14
    Map[[22, 39, 40, 86, 87]] = 5;
    Map[[27, 29, 44, 78, 93]] = 15
    Map[[5, 20, 25, 84, 94]] = 6;
    Map[[36, 50, 65, 74, 80]] = 16
    Map[[6, 7, 14, 18, 24]] = 7;
    Map[[47, 52, 56, 59, 96]] = 17
    Map[[3, 42, 43, 88, 97]] = 8;
    Map[[8, 13, 48, 58, 90]] = 18
    Map[[12, 17, 37, 68, 76]] = 9;
    Map[[41, 69, 81, 85, 89]] = 19

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

print("Loading data..." + '\n')

# trainX = image data ----- trainF = fine labels ----- trainC = coarse labels
(trainX, trainF), (testX, testF) = cifar100.load_data(label_mode = 'fine')
(_, trainC), (_, testC) = cifar100.load_data(label_mode = 'coarse')

dir = "C://Users//Cameron//PycharmProjects//EECS545_Project//"
meta = unpickle(dir + "meta")
fineLabels = meta[b'fine_label_names']
coarseLabels = meta[b'coarse_label_names']

print(labelMap())

print("Labels: {}".format(trainF[:5]))
print("One-hot Labels: {}".format(to_categorical(trainF)[:5,:]))
print("Get labels back: {}".format(np.argmax(to_categorical(trainF)[:5,:], axis=1)))

for i in range(trainX.shape[0]):
    image = trainX[i, :, :, :].reshape((32,32,3))
    print("Class: {}, {}".format(trainF[i][0], fineLabels[trainF[i][0]]))
    print("Superclass: {}, {}".format(trainC[i][0], coarseLabels[trainC[i][0]]))
    cv2.imshow("image", image)
    cv2.waitKey(0)

