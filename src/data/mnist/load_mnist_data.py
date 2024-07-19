from keras.datasets import mnist  # to download mnist dataset


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_train = x_train.astype("float32")/255.
    x_test = x_test.astype("float32")/255.
    return (x_train, y_train), (x_test, y_test)