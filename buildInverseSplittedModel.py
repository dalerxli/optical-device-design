import tensorflow as tf


'''
author: Yuan Liu
University: Southeast University
'''


class GNP_Network():

    def __init__(self, parallel_unit):

        self.input_x_polarity = tf.placeholder(tf.float32, [None, 43], "x_polarity")
        self.input_y_polarity = tf.placeholder(tf.float32, [None, 43], "y_polarity")
        self.material = tf.placeholder(tf.float32, [None, 25], "material_properties")
        self.geometries = tf.placeholder(tf.float32, [None, 8], "labels")
        self.training = tf.placeholder(tf.bool, [], "training")

        with tf.name_scope("x_polarity_parallel"):
            self.x_b_1 = tf.get_variable("x_b_1", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.x_b_2 = tf.get_variable("x_b_2", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.x_b_3 = tf.get_variable("x_b_3", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.x_w_1 = tf.get_variable("x_w_1", [43, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.x_w_2 = tf.get_variable("x_w_2", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.x_w_3 = tf.get_variable("x_w_3", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            x_output_1 = tf.nn.xw_plus_b(self.input_x_polarity, self.x_w_1, self.x_b_1)
            x_output_2 = tf.nn.xw_plus_b(x_output_1, self.x_w_2, self.x_b_2)
            x_output_3 = tf.nn.xw_plus_b(x_output_2, self.x_w_3, self.x_b_3)

        with tf.name_scope("y_polarity_parallel"):
            self.y_b_1 = tf.get_variable("y_b_1", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.y_b_2 = tf.get_variable("y_b_2", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.y_b_3 = tf.get_variable("y_b_3", [parallel_unit], tf.float32, initializer=tf.constant_initializer(0.0))
            self.y_w_1 = tf.get_variable("y_w_1", [43, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.y_w_2 = tf.get_variable("y_w_2", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.y_w_3 = tf.get_variable("y_w_3", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            y_output_1 = tf.nn.xw_plus_b(self.input_y_polarity, self.y_w_1, self.y_b_1)
            y_output_2 = tf.nn.xw_plus_b(y_output_1, self.y_w_2, self.y_b_2)
            y_output_3 = tf.nn.xw_plus_b(y_output_2, self.y_w_3, self.y_b_3)

        with tf.name_scope("material_properties"):
            self.m_b_1 = tf.get_variable("m_b_1", [parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.m_b_2 = tf.get_variable("m_b_2", [parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.m_b_3 = tf.get_variable("m_b_3", [parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.m_w_1 = tf.get_variable("m_w_1", [25, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.m_w_2 = tf.get_variable("m_w_2", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.m_w_3 = tf.get_variable("m_w_3", [parallel_unit, parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            m_output_1 = tf.nn.xw_plus_b(self.material, self.m_w_1, self.m_b_1)
            m_output_2 = tf.nn.xw_plus_b(m_output_1, self.m_w_2, self.m_b_2)
            m_output_3 = tf.nn.xw_plus_b(m_output_2, self.m_w_3, self.m_b_3)

        with tf.name_scope("full_connected"):
            all_neurals = tf.concat([x_output_3, y_output_3, m_output_3], -1)
            self.f_w_1 = tf.get_variable("f_w_1", [3*parallel_unit, 3*parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_2 = tf.get_variable("f_w_2", [3 * parallel_unit, 3 * parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_3 = tf.get_variable("f_w_3", [3 * parallel_unit, 3 * parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_4 = tf.get_variable("f_w_4", [3 * parallel_unit, 3 * parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_5 = tf.get_variable("f_w_5", [3 * parallel_unit, 3 * parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_6 = tf.get_variable("f_w_6", [3 * parallel_unit, 3 * parallel_unit], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_w_7 = tf.get_variable("f_w_7", [3 * parallel_unit, 8], tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            self.f_b_1 = tf.get_variable("f_b_1", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_2 = tf.get_variable("f_b_2", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_3 = tf.get_variable("f_b_3", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_4 = tf.get_variable("f_b_4", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_5 = tf.get_variable("f_b_5", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_6 = tf.get_variable("f_b_6", [parallel_unit*3], tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.f_b_7 = tf.get_variable("f_b_7", [8], tf.float32, initializer=tf.constant_initializer(0.0))

            f_output_1 = tf.nn.xw_plus_b(all_neurals, self.f_w_1, self.f_b_1)
            f_output_1_relu = tf.nn.relu(f_output_1)
            y_1 = tf.layers.batch_normalization(f_output_1_relu, training=self.training)
            f_output_2 = tf.nn.xw_plus_b(y_1, self.f_w_2, self.f_b_2)
            f_output_2_relu = tf.nn.relu(f_output_2)
            y_2 = tf.layers.batch_normalization(f_output_2_relu, training=self.training)
            f_output_3 = tf.nn.xw_plus_b(y_2, self.f_w_3, self.f_b_3)
            f_output_3_relu = tf.nn.relu(f_output_3)
            y_2 = tf.layers.batch_normalization(f_output_3_relu, training=self.training)
            f_output_4 = tf.nn.xw_plus_b(y_2, self.f_w_4, self.f_b_4)
            f_output_4_relu = tf.nn.relu(f_output_4)
            y_4 = tf.layers.batch_normalization(f_output_4_relu, training=self.training)
            f_output_5 = tf.nn.xw_plus_b(y_4, self.f_w_5, self.f_b_5)
            f_output_5_relu = tf.nn.relu(f_output_5)
            y_5 = tf.layers.batch_normalization(f_output_5_relu, training=self.training)
            f_output_6 = tf.nn.xw_plus_b(y_5, self.f_w_6, self.f_b_6)
            f_output_6_relu = tf.nn.relu(f_output_6)
            y_6 = tf.layers.batch_normalization(f_output_6_relu, training=self.training)
            self.scores = tf.nn.xw_plus_b(y_6, self.f_w_7, self.f_b_7)

        with tf.name_scope("loss"):
            self.losses = tf.losses.mean_squared_error(self.geometries, self.scores)

    def return_variables(self):

        return [self.x_b_1, self.x_b_2, self.x_b_3, self.x_w_1, self.x_w_2, self.x_w_3, self.y_b_1, self.y_b_2,
                self.y_b_3, self.y_w_1, self.y_w_2, self.y_w_3, self.m_b_1, self.m_b_2, self.m_b_3, self.m_w_1,
                self.m_w_2, self.m_w_3, self.f_b_1, self.f_b_2, self.f_b_3, self.f_b_4, self.f_b_5, self.f_b_6,
                self.f_b_7, self.f_w_1, self.f_w_2, self.f_w_3, self.f_w_4, self.f_w_5, self.f_w_6, self.f_w_7]









