"""
Tensorflow MLP optimizers benchmark with the same model:

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
increment_step = global_step.assign_add(1)

"""


import numpy as np
import tensorflow as tf


def tf_optimizer(optimize = "adam", learning_rate = 0.001):
    if optimize == 'adadelta':
       optimizer = tf.train.AdadeltaOptimizer(
           learning_rate)
    elif optimize == 'adagrad':
       optimizer = tf.train.AdagradOptimizer(
           learning_rate)
    elif optimize == 'adam':
       optimizer = tf.train.AdamOptimizer(
           learning_rate)
    elif optimize == 'ftrl':
       optimizer = tf.train.FtrlOptimizer(
           learning_rate)
    elif optimize == 'momentum':
       optimizer = tf.train.MomentumOptimizer(
           learning_rate,
           momentum = 0.9,
           name='Momentum')
    elif optimize == 'rmsprop':
       optimizer = tf.train.RMSPropOptimizer(
           learning_rate)
    elif optimize == 'sgd':
       optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer



class BO():
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        """The class builds a MLP network and trains a model with the input data



        """
        self.x_train = x_train / 255.0
        self.y_train =y_train
        self.x_valid = x_valid / 255.0
        self.y_valid = y_valid
        self.x_test = x_test / 255.0
        self.y_test = y_test
        self.input = tf.placeholder("float", [None, 32, 32, 3])
        self.output = tf.placeholder("float", [None, self.y_train.shape[1]])
        self.layers = {}
        self.train_step = None

    def build_graph(self):
        self.layers["layer1"] = tf.contrib.layers.conv2d(self.input, 64, 5,  1, padding = "SAME", activation_fn = tf.nn.relu, scope="conv1")
        self.layers["layer2"] = tf.contrib.layers.max_pool2d(self.layers["layer1"], kernel_size=[3, 3], stride=[2, 2], padding = "SAME", scope="pool2" )
        self.layers["layer3"] = tf.nn.lrn(self.layers["layer2"], 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        self.layers["layer4"] = tf.contrib.layers.conv2d(self.layers["layer3"], 64, 5,  1, padding = "SAME", activation_fn = tf.nn.relu, scope="conv4")
        self.layers["layer5"] = tf.nn.lrn(self.layers["layer4"], 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')
        self.layers["layer6"] = tf.contrib.layers.max_pool2d(self.layers["layer5"], kernel_size=[3, 3], stride=[2, 2], padding = "SAME", scope = "pool6")
        shape = self.layers["layer6"].get_shape().as_list()
        self.layers["layer7"] = tf.contrib.layers.fully_connected(tf.reshape(self.layers["layer6"],[-1,shape[1]*shape[2]*shape[3]]), num_outputs = 384, activation_fn = tf.nn.relu, scope = "fc7")
        self.layers["layer8"] =tf.contrib.layers.fully_connected(self.layers["layer7"], num_outputs = 192, activation_fn = tf.nn.relu, scope = "fc8")
        self.layers["layer9"] = tf.contrib.layers.fully_connected(self.layers["layer8"], num_outputs = self.y_train.shape[1], activation_fn = tf.nn.softmax, scope = "output")
        return self


    def compile_graph(self, optimize = "adam", learning_rate = 0.001):
        print ("[Using optimizer]:", optimize)
        with tf.name_scope("cross_entropy"):
            self._output = tf.reduce_mean(-tf.reduce_sum(self.output * tf.log(self.layers["layer9"]+0.00001),[1]))
        tf.summary.scalar("cross_entropy", self._output)

        with tf.name_scope("accuracy"):
            self._prediction = tf.equal(tf.argmax(self.layers["layer9"], 1), tf.argmax(self.output,1))
            self._accuracy = tf.reduce_mean(tf.cast(self._prediction, tf.float32))
        tf.summary.scalar("accuracy", self._accuracy)

        with tf.name_scope("gradient_calculations"):
            optimizer = tf_optimizer(optimize = optimize, learning_rate = learning_rate)
            self.train_step = optimizer.minimize(self._output)
        return self


    def train(self,  batch_size = 32, epochs = 100, summary_dir = "/tmp/mnist"):
        # saver = tf.train.Saver(tf.global_variables())
        print("[Training started]")
        merged = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            training_writer = tf.summary.FileWriter(summary_dir + '/training',sess.graph)
            train_writer = tf.summary.FileWriter(summary_dir+'/train')
            valid_writer = tf.summary.FileWriter(summary_dir + '/valid')

            iterations = int(self.x_train.shape[0]/batch_size)
            for ep in range(epochs):
                for i in range(iterations):
                    batch_x , batch_y = self.x_train[i*batch_size:(i+1)*batch_size], self.y_train[i*batch_size:(i+1)*batch_size]
                    summary, _, c = sess.run([merged , self.train_step, self._output ], feed_dict = {self.input: batch_x, self.output: batch_y})
                    training_writer.add_summary(summary, ep*iterations+i)
                    if i%100 ==0:
                        with tf.device("/cpu:0"):
                            train_cost, train_acc = sess.run([self._output, self._accuracy], feed_dict={self.input: self.x_train[0:5000], self.output: self.y_train[0:5000]})
                        train_writer.add_summary(summary, ep*iterations+i)
                        with tf.device("/cpu:0"):
                            valid_cost, valid_acc = sess.run([self._output, self._accuracy], feed_dict={self.input: self.x_valid, self.output: self.y_valid})
                        valid_writer.add_summary(summary, ep*iterations+i)
                print ("epoch: ", ep, "train_cost: ", train_cost , "Train_Accuracy: ", train_acc, "valid_cost: ", valid_cost, "Validation_Accuracy: ", valid_acc)
            print("optimization_finished")
            test_accuracy = sess.run([self._accuracy],feed_dict={self.input: self.x_test, self.output: self.y_test} )
            print("Test Accuracy:", test_accuracy)
