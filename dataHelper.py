import tensorflow as tf
import numpy as np

'''
author: Yuan Liu
University: Southeast University
'''


def dataset_input_fn(fun):

    def wrapper(add_x, add_y):
        data_x, data_y, test_x, test_y = fun(add_x, add_y)
        dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
        return [dataset, test_x, test_y]
    return wrapper


@dataset_input_fn
def normalize_data(filename_x, filename_y):
    x = np.load(filename_x)
    y = np.load(filename_y)
    assert x.shape[0] == y.shape[0]
    for i in range(x.shape[1]):
        x_mean = np.mean(x[:, i])
        x_std = np.std(x[:, i])
        if x_std == 0:
            x_std = 1
        x[:, i] = (x[:, i] - x_mean)/x_std
    length = len(x)
    x_train = x[: -int(0.1*length)]
    y_train = y[: -int(0.1*length)]
    x_test = x[-int(0.1*length):]
    y_test = y[-int(0.1*length):]
    return [x_train, y_train, x_test, y_test]


if __name__ == "__main__":
    add_x = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\trainData.npy"
    add_y = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\trainLabel.npy"
    batched_dataset = normalize_data(add_x, add_y).batch(4)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                sess.run(iterator.initializer)
                x, y = sess.run(next_element)
                material_prop = x[:, :25]
                polarity_x = x[:, 25:68]
                polarity_y = x[:, 68:]
            except tf.errors.OutOfRangeError:
                pass


def test_normalize(data):

    for i in range(data.shape[1]):
        x_mean = np.mean(data[:, i])
        x_std = np.std(data[:, i])
        if x_std == 0:
            x_std = 1
        data[:, i] = (data[:, i] - x_mean) / x_std

    return data








