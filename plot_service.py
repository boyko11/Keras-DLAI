import numpy as np
import matplotlib.pyplot as plt


def plot_training_image(img_idx, x_train, y_train):

    plt.imshow(x_train[img_idx])
    plt.show()
    print("y = " + str(np.squeeze(y_train[:, img_idx])))
