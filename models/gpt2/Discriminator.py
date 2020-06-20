import tensorflow as tf
import numpy as np
from models.Gens import Dis


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(Dis):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size, discriminator_name,
            emd_dim, filter_sizes, num_filters, l2_reg_lambda=0.0,dis_lr=1e-3):
        # Placeholders for input, output and dropout
        self.dis_lr = dis_lr
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.sequence_length = sequence_length

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.scores = tf.nn.softmax(self.logits)

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Calculate the dd of each batch,and record
            self.distance_eval_info()

        self.params = [param for param in tf.trainable_variables() if discriminator_name in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.dis_lr)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def distance_eval_info(self):
        # true and fake samples number
        self.true_fake_num = [tf.cast(tf.count_nonzero(self.input_y[:, 1]), tf.float32),
                              tf.cast(tf.count_nonzero(self.input_y[:, 0]), tf.float32)]

        self.true_samples_score = self.scores[:, 1] * self.input_y[:, 1]
        self.fake_samples_score = self.scores[:, 1] * self.input_y[:, 0]
        self.true_average = tf.reduce_sum(self.true_samples_score) / self.true_fake_num[0]
        self.fake_average = tf.reduce_sum(self.fake_samples_score) / self.true_fake_num[1]
        self.true_sum = tf.reduce_sum(self.true_samples_score)
        self.fake_sum = tf.reduce_sum(self.fake_samples_score)

        self.true_greater_index = tf.cast(tf.greater(self.true_samples_score, 0.5), tf.float32)
        self.true_less_index = tf.cast(tf.less_equal(self.true_samples_score, 0.5), tf.float32)
        self.fake_greater_index = tf.cast(tf.greater(self.fake_samples_score, 0.5), tf.float32)
        self.fake_less_index = tf.cast(tf.less_equal(self.fake_samples_score, 0.5), tf.float32)

        self.true_correct = self.true_greater_index * 1.0
        self.true_error = self.true_less_index * self.input_y[:, 1]
        self.fake_error = self.fake_greater_index * 1.0
        self.fake_correct = self.fake_less_index * self.input_y[:, 0]

        ############
        # tensorboard infos batch
        ############
        # train-info
        batch_train_loss = tf.summary.scalar('batch_train_loss', self.loss)
        batch_train_true_score = tf.summary.scalar('batch_train_true_score', self.true_average)
        batch_train_fake_score = tf.summary.scalar('batch_train_fake_score', self.fake_average)

        # valid-info
        batch_valid_loss = tf.summary.scalar('batch_valid_loss', self.loss)
        batch_valid_true_score = tf.summary.scalar('batch_valid_true_score', self.true_average)
        batch_valid_fake_score = tf.summary.scalar('batch_valid_fake_score', self.fake_average)

        self.merge_summary_train_batch = tf.summary.merge(
            [batch_train_loss, batch_train_true_score, batch_train_fake_score])
        self.merge_summary_valid_batch = tf.summary.merge(
            [batch_valid_loss, batch_valid_true_score, batch_valid_fake_score])

        ############
        # tensorboard infos epoch
        ############
        self.train_acc = tf.placeholder(tf.float32)
        self.train_dd = tf.placeholder(tf.float32)
        self.train_loss = tf.placeholder(tf.float32)
        self.train_t_avr = tf.placeholder(tf.float32)
        self.train_f_avr = tf.placeholder(tf.float32)

        self.valid_acc = tf.placeholder(tf.float32)
        self.valid_dd = tf.placeholder(tf.float32)
        self.valid_loss = tf.placeholder(tf.float32)
        self.valid_t_avr = tf.placeholder(tf.float32)
        self.valid_f_avr = tf.placeholder(tf.float32)

        self.test_acc = tf.placeholder(tf.float32)
        self.test_dd = tf.placeholder(tf.float32)
        self.test_loss = tf.placeholder(tf.float32)
        self.test_t_avr = tf.placeholder(tf.float32)
        self.test_f_avr = tf.placeholder(tf.float32)

        train_acc_all = tf.summary.scalar('train_accuracy', self.train_acc)
        train_dd_all = tf.summary.scalar('train_dd', self.train_dd)
        train_loss_all = tf.summary.scalar('train_loss', self.train_loss)
        train_truescore_avr = tf.summary.scalar('train_truescore_avr', self.train_t_avr)
        train_fakescore_avr = tf.summary.scalar('train_fakescore_avr', self.train_f_avr)

        valid_acc_all = tf.summary.scalar('valid_accuracy', self.valid_acc)
        valid_dd_all = tf.summary.scalar('valid_dd', self.valid_dd)
        valid_loss_all = tf.summary.scalar('valid_loss', self.valid_loss)
        valid_truescore_avr = tf.summary.scalar('valid_truescore_avr', self.valid_t_avr)
        valid_fakescore_avr = tf.summary.scalar('valid_fakescore_avr', self.valid_f_avr)

        test_acc_all = tf.summary.scalar('test_accuracy', self.test_acc)
        test_dd_all = tf.summary.scalar('test_dd', self.test_dd)
        test_loss_all = tf.summary.scalar('test_loss', self.test_loss)
        test_truescore_avr = tf.summary.scalar('test_truescore_avr', self.test_t_avr)
        test_fakescore_avr = tf.summary.scalar('test_fakescore_avr', self.test_f_avr)

        self.merge_train = tf.summary.merge(
            [train_acc_all, train_dd_all, train_loss_all, train_truescore_avr, train_fakescore_avr])
        self.merge_valid = tf.summary.merge(
            [valid_acc_all, valid_dd_all, valid_loss_all, valid_truescore_avr, valid_fakescore_avr])
        self.merge_test = tf.summary.merge(
            [test_acc_all, test_dd_all, test_loss_all, test_truescore_avr, test_fakescore_avr])
