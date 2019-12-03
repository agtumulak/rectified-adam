import numpy as np
import matplotlib.pyplot as plt

labels = np.array([0.01, 0.001, 0.0001, 0.00001, 0.000001])
accuracies = np.arange(1,16).reshape(5,3).astype('float32')
x = np.arange(5)
width = 0.28

rects1 = plt.bar(x-width, accuracies[:,0], width, color='r')
rects2 = plt.bar(x, accuracies[:,1], width, color='b')
rects3 = plt.bar(x+width, accuracies[:,2], width, color='y')
bars = [rects1, rects2, rects3]

plt.xticks(x, labels)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.legend(("Fine Labels", "Fine Labels (Top 3)", "Coarse Labels"), loc="upper left")
plt.title("RAdam Accuracy Metrics for Varying Learning Rates")

for rects in bars:
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.show()