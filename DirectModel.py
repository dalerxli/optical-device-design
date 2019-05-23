import tensorflow as tf


'''
author: Yuan Liu
University: Southeast University
'''

class SPN_Network_x():

    def __init__(self, predict_ge):

        with tf.name_scope("spn_x"):
            self.material_spn = tf.placeholder(tf.float32, [None, 25], "material_spn")
            self.flags = tf.placeholder(tf.float32, [None, 1], "flags_x")
            self.concat_data = tf.concat([predict_ge, self.flags, self.material_spn], -1)
            self.x_spectrum = tf.placeholder(tf.float32, [None, 43], "x_spectrum")
            self.training = tf.placeholder(tf.bool, [], "training")
            self.spn_w_1 = tf.get_variable("spn_w_1_x", [8+1+25, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_2 = tf.get_variable("spn_w_2_x", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_3 = tf.get_variable("spn_w_3_x", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_4 = tf.get_variable("spn_w_4_x", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_5 = tf.get_variable("spn_w_5_x", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_6 = tf.get_variable("spn_w_6_x", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_7 = tf.get_variable("spn_w_7_x", [1000, 43], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_1 = tf.get_variable("spn_b_1_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_2 = tf.get_variable("spn_b_2_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_3 = tf.get_variable("spn_b_3_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_4 = tf.get_variable("spn_b_4_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_5 = tf.get_variable("spn_b_5_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_6 = tf.get_variable("spn_b_6_x", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_7 = tf.get_variable("spn_b_7_x", [43], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            spn_output_1 = tf.nn.xw_plus_b(self.concat_data, self.spn_w_1, self.spn_b_1)
            spn_output_1_relu = tf.nn.relu(spn_output_1)
            spn_y_1 = tf.layers.batch_normalization(spn_output_1_relu, training=self.training)
            spn_output_2 = tf.nn.xw_plus_b(spn_y_1, self.spn_w_2, self.spn_b_2)
            spn_output_2_relu = tf.nn.relu(spn_output_2)
            spn_y_2 = tf.layers.batch_normalization(spn_output_2_relu, training=self.training)
            spn_output_3 = tf.nn.xw_plus_b(spn_y_2, self.spn_w_3, self.spn_b_3)
            spn_output_3_relu = tf.nn.relu(spn_output_3)
            spn_y_3 = tf.layers.batch_normalization(spn_output_3_relu, training=self.training)
            spn_output_4 = tf.nn.xw_plus_b(spn_y_3, self.spn_w_4, self.spn_b_4)
            spn_output_4_relu = tf.nn.relu(spn_output_4)
            spn_y_4 = tf.layers.batch_normalization(spn_output_4_relu, training=self.training)
            spn_output_5 = tf.nn.xw_plus_b(spn_y_4, self.spn_w_5, self.spn_b_5)
            spn_output_5_relu = tf.nn.relu(spn_output_5)
            spn_y_5 = tf.layers.batch_normalization(spn_output_5_relu, training=self.training)
            spn_output_6 = tf.nn.xw_plus_b(spn_y_5, self.spn_w_6, self.spn_b_6)
            spn_output_6_relu = tf.nn.relu(spn_output_6)
            spn_y_6 = tf.layers.batch_normalization(spn_output_6_relu, training=self.training)
            self.scores = tf.nn.xw_plus_b(spn_y_6, self.spn_w_7, self.spn_b_7)

        with tf.name_scope("losses"):
            self.losses = tf.losses.mean_squared_error(self.x_spectrum, self.scores)

    def return_variable(self):
        return [self.spn_w_1, self.spn_w_2, self.spn_w_3, self.spn_w_4, self.spn_w_5, self.spn_w_6,
                self.spn_w_7, self.spn_b_1, self.spn_b_2, self.spn_b_3, self.spn_b_4, self.spn_b_5,
                self.spn_b_6, self.spn_b_7]


class SPN_Network_y():

    def __init__(self, predict_ge):

        with tf.name_scope("spn_y"):
            self.material_spn = tf.placeholder(tf.float32, [None, 25], "material_spn")
            self.flags = tf.placeholder(tf.float32, [None, 1], "flags_y")
            self.concat_data = tf.concat([predict_ge, self.flags, self.material_spn], -1)
            self.y_spectrum = tf.placeholder(tf.float32, [None, 43], "y_spectrum")
            self.training = tf.placeholder(tf.bool, [], "training")
            self.spn_w_1 = tf.get_variable("spn_w_1_y", [8 + 1 + 25, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_2 = tf.get_variable("spn_w_2_y", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_3 = tf.get_variable("spn_w_3_y", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_4 = tf.get_variable("spn_w_4_y", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_5 = tf.get_variable("spn_w_5_y", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_6 = tf.get_variable("spn_w_6_y", [1000, 1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_w_7 = tf.get_variable("spn_w_7_y", [1000, 43], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_1 = tf.get_variable("spn_b_1_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_2 = tf.get_variable("spn_b_2_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_3 = tf.get_variable("spn_b_3_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_4 = tf.get_variable("spn_b_4_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_5 = tf.get_variable("spn_b_5_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_6 = tf.get_variable("spn_b_6_y", [1000], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.spn_b_7 = tf.get_variable("spn_b_7_y", [43], tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            spn_output_1 = tf.nn.xw_plus_b(self.concat_data, self.spn_w_1, self.spn_b_1)
            spn_output_1_relu = tf.nn.relu(spn_output_1)
            spn_y_1 = tf.layers.batch_normalization(spn_output_1_relu, training=self.training)
            spn_output_2 = tf.nn.xw_plus_b(spn_y_1, self.spn_w_2, self.spn_b_2)
            spn_output_2_relu = tf.nn.relu(spn_output_2)
            spn_y_2 = tf.layers.batch_normalization(spn_output_2_relu, training=self.training)
            spn_output_3 = tf.nn.xw_plus_b(spn_y_2, self.spn_w_3, self.spn_b_3)
            spn_output_3_relu = tf.nn.relu(spn_output_3)
            spn_y_3 = tf.layers.batch_normalization(spn_output_3_relu, training=self.training)
            spn_output_4 = tf.nn.xw_plus_b(spn_y_3, self.spn_w_4, self.spn_b_4)
            spn_output_4_relu = tf.nn.relu(spn_output_4)
            spn_y_4 = tf.layers.batch_normalization(spn_output_4_relu, training=self.training)
            spn_output_5 = tf.nn.xw_plus_b(spn_y_4, self.spn_w_5, self.spn_b_5)
            spn_output_5_relu = tf.nn.relu(spn_output_5)
            spn_y_5 = tf.layers.batch_normalization(spn_output_5_relu, training=self.training)
            spn_output_6 = tf.nn.xw_plus_b(spn_y_5, self.spn_w_6, self.spn_b_6)
            spn_output_6_relu = tf.nn.relu(spn_output_6)
            spn_y_6 = tf.layers.batch_normalization(spn_output_6_relu, training=self.training)
            self.scores = tf.nn.xw_plus_b(spn_y_6, self.spn_w_7, self.spn_b_7)

        with tf.name_scope("losses"):
            self.losses = tf.losses.mean_squared_error(self.y_spectrum, self.scores)

    def return_variable(self):

        return [self.spn_w_1, self.spn_w_2, self.spn_w_3, self.spn_w_4, self.spn_w_5, self.spn_w_6,
                self.spn_w_7, self.spn_b_1, self.spn_b_2, self.spn_b_3, self.spn_b_4, self.spn_b_5,
                self.spn_b_6, self.spn_b_7]


