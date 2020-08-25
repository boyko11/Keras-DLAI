import keras.backend as K
import data_service, plot_service
from happy_model import HappyModel
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


K.set_image_data_format('channels_last')

X_train, Y_train, X_test, Y_test, classes = data_service.load_dataset()
plot_service.plot_training_image(2, X_train, Y_train)

X_train, Y_train, X_test, Y_test, = data_service.norm_and_reshape(X_train, Y_train, X_test, Y_test,)


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

m, rows, cols, channels = X_train.shape
happyModel = HappyModel((rows, cols, channels))
happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

happyModel.fit(X_train, Y_train, batch_size=16, epochs=3)

preds = happyModel.predict(X_train)
print()
print ("Train Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))

preds = happyModel.predict(X_test)
print()
print ("Test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

happyModel.summary()

plot_model(happyModel, to_file='plots/HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))









