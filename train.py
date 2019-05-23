from buildInverseSplittedModel import GNP_Network
import tensorflow as tf
import time
import os
import datetime
from dataHelper import normalize_data
import numpy as np
from DirectModel import SPN_Network_x, SPN_Network_y

'''
author: Yuan Liu
University: Southeast University

'''


def train(train_data, train_label, test_data, test_label, epochs, batch_size):

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess = tf.InteractiveSession(config=session_config)
    GPN = GNP_Network(250)
    SPN_x = SPN_Network_x(GPN.scores)
    SPN_y = SPN_Network_y(GPN.scores)
    global_step = tf.get_variable("global_step", [], tf.int64, initializer=tf.constant_initializer(0))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # spn_x_train_op = tf.train.AdamOptimizer().minimize(SPN_x.losses, global_step=global_step,
    #                                                    var_list=SPN_x.return_variable())
    # spn_x_train_op = tf.group([update_ops, spn_x_train_op])
    # spn_y_train_op = tf.train.AdamOptimizer().minimize(SPN_y.losses, var_list=SPN_y.return_variable())
    # spn_y_train_op = tf.group([update_ops, spn_y_train_op])
    total_losses = GPN.losses + SPN_x.losses + SPN_x.losses
    total_train_op = tf.train.AdamOptimizer().minimize(total_losses, global_step=global_step)
    total_train_op = tf.group([update_ops, total_train_op])
    time_stamp = str(int(time.time()))
    out_dir = os.path.join(os.curdir, time_stamp)
    print("Writing to: {}".format(out_dir))

    loss_summary = tf.summary.scalar("losses", total_losses)

    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    test_summary_op = tf.summary.merge([loss_summary])
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    def train_step(x_p, y_p, m, label):
        flags_x = np.ones((np.shape(x_p)[0], 1))
        flags_y = np.zeros((np.shape(x_p)[0], 1))
        feed_dict = {
            GPN.input_x_polarity: x_p,
            GPN.input_y_polarity: y_p,
            GPN.material: m,
            GPN.geometries: label,
            GPN.training: True,
            SPN_x.material_spn: m,
            SPN_x.x_spectrum: x_p,
            SPN_x.flags: flags_x,
            SPN_x.training: True,
            SPN_y.material_spn: m,
            SPN_y.y_spectrum: y_p,
            SPN_y.flags: flags_y,
            SPN_y.training: True
        }

        _, step, summary, loss = sess.run([total_train_op, global_step, train_summary_op, total_losses], feed_dict)
        # if global_step % 20 == 0:
        #     _, step, summary, loss = sess.run([total_train_op, global_step, train_summary_op, total_losses],
        #                                       feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{} step: {}, loss: {}".format(time_str, step, loss))
        train_summary_writer.add_summary(summary, step)

    def test_step(x_p, y_p, m, label):
        flags_x = np.ones((np.shape(x_p)[0], 1))
        flags_y = np.zeros((np.shape(x_p)[0], 1))
        feed_dict = {
            GPN.input_x_polarity: x_p,
            GPN.input_y_polarity: y_p,
            GPN.material: m,
            GPN.geometries: label,
            GPN.training: False,
            SPN_x.material_spn: m,
            SPN_x.x_spectrum: x_p,
            SPN_x.flags: flags_x,
            SPN_x.training: False,
            SPN_y.material_spn: m,
            SPN_y.y_spectrum: y_p,
            SPN_y.flags: flags_y,
            SPN_y.training: False
        }

        step, summary, loss = sess.run([global_step, test_summary_op, total_losses], feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{} step: {}, loss: {}".format(time_str, step, loss))
        test_summary_writer.add_summary(summary, step)

    batched_dataset, test_x, test_y = normalize_data(train_data, train_label)
    iterator = batched_dataset.batch(batch_size).make_initializable_iterator()
    next_element = iterator.get_next()
    # test_x = np.load(test_data)
    # test_y = np.load(test_label)
    # test_x = test_normalize(test_x)
    # test_y = test_normalize(test_y)
    test_material = test_x[:, :25]
    test_p_x = test_x[:, 25:68]
    test_p_y = test_x[:, 68:]

    sess.run(tf.global_variables_initializer())
    for _ in range(epochs):

        sess.run(iterator.initializer)
        while True:
            try:
                x, y = sess.run(next_element)
                material_prop = x[:, :25]
                polarity_x = x[:, 25:68]
                polarity_y = x[:, 68:]
                train_step(polarity_x, polarity_y, material_prop, y)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    test_step(test_p_x, test_p_y, test_material, test_y)

            except tf.errors.OutOfRangeError:
                break


def main(argv=None):
    train_x_file = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\trainData.npy"
    train_y_file = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\trainLabel.npy"
    test_x_file = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\testData.npy"
    test_y_file = r"C:\Users\liuyuan\Desktop\trainData\numpy_data\testLabel.npy"
    train(train_x_file, train_y_file, test_x_file, test_y_file, 2000, 64)


if __name__ == "__main__":
    tf.app.run()






