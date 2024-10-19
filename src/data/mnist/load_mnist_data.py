from keras.datasets import mnist  # to download mnist dataset


def load_mnist_data():
    """
    Load mnist train/test dataset from keras library and normalize it to [0,1] values
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32")/255.  # normalize to [0,1] values
    x_test = x_test.astype("float32")/255.    # normalize to [0,1] values

    return (x_train, y_train), (x_test, y_test)