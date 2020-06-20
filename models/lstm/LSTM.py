import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
from models.Gens import Gen


class Generator(Gen):
    def __init__(self, num_vocabulary, batch_size, hidden_dim, sequence_length, generator_name,
                 learning_rate=0.001, grad_clip=10.0, dropout_keep_prob=0.5, start_token=2, num_layers_gen=2):
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.reward_gamma = 0.95
        self.grad_clip = grad_clip
        self.num_layers_gen = num_layers_gen
        self.run_keep_drop = dropout_keep_prob
        self.temperature = tf.placeholder(tf.float32)

        # tensor placeholder
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
            self.input_mask = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length])
            self.dropout_keep_place = tf.placeholder(tf.float32)

            self.embedding = tf.get_variable("embedding", [self.num_vocabulary, self.hidden_dim])
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            inputs = tf.nn.dropout(inputs, self.dropout_keep_place)

        def get_cell(hidden_dim, keep_prob):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        with tf.variable_scope("rnns"):
            cells = tf.contrib.rnn.MultiRNNCell([get_cell(self.hidden_dim, self.dropout_keep_place) for _ in range(self.num_layers_gen)],
                                                state_is_tuple=True)
            self.initial_state = cells.zero_state(self.batch_size, tf.float32)  
            self.initial_state_gen_x = cells.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        state_gen_x = self.initial_state_gen_x
        gen_x = []
        self.gen_x_batch = []

        weight = tf.get_variable("weight", [self.hidden_dim, self.num_vocabulary])
        bias = tf.get_variable("bias", [self.num_vocabulary])
        with tf.variable_scope("generator"):

            for time_step in range(self.sequence_length):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cells(inputs[:, time_step, :], state)
                outputs.append(cell_output)

            for t in range(self.sequence_length):
                tf.get_variable_scope().reuse_variables()
                if t > 0:
                    gen_in = tf.nn.embedding_lookup(self.embedding, gen_x_tmp)
                else:
                    gen_in = inputs
                cell_output_gen_x, state_gen_x = cells(gen_in[:, 0, :], state_gen_x)
                logit_step = tf.matmul(cell_output_gen_x, weight) + bias
                prob_step = tf.nn.softmax(logit_step * self.temperature)
                gen_x_tmp = tf.cast(tf.multinomial(tf.log(prob_step), 1), tf.int32) 
                gen_x.append(gen_x_tmp)
            self.gen_x_batch = tf.concat(gen_x, 1)

        output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_dim])
        logits = tf.matmul(output, weight) + bias 
        self.score = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.sequence_length, self.num_vocabulary])
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.sequence_length], dtype=tf.float32)])

        self.eval_losses = tf.reshape(loss, [self.batch_size, self.sequence_length])
        self.cost = tf.reduce_sum(loss) / (self.sequence_length*self.batch_size)
        self.final_state = state

        self.params = [param for param in tf.trainable_variables() if generator_name in param.name]

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.params), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, self.params))

        g_train_loss =  tf.summary.scalar('g_train_loss', self.cost)
        self.merge_summary_train = tf.summary.merge([g_train_loss])

    def run_epoch(self, sess, data_loader, writer, epoch):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            y, mask = data_loader.next_batch()
            
            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length-1]

            cost, _, _, merge_summary_train = sess.run([self.cost, self.final_state, self.train_op,self.merge_summary_train],
                                        {self.input_data: x, self.targets: y, self.input_mask:mask, self.dropout_keep_place:self.run_keep_drop})
            
            writer.add_summary(merge_summary_train, epoch*data_loader.num_batch+it)
            total_costs += cost

        nll_avr = total_costs/data_loader.num_batch
        return nll_avr

    def eval_epoch(self, sess, data_loader):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            y, mask = data_loader.next_batch()
            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length-1]

            costs, _= sess.run([self.eval_losses, self.final_state],
                                        {self.input_data: x, self.targets: y, self.input_mask:mask, self.dropout_keep_place:1.0})
                        
            total_costs += np.sum(costs * mask) / np.sum(mask)

        nll_avr = total_costs/data_loader.num_batch
        return nll_avr

    def generate(self, sess, t):
        input = np.ones((self.batch_size, 1), dtype=int)*2
        outputs = sess.run(self.gen_x_batch, {self.input_data:input,
                                              self.dropout_keep_place:1.0,
                                              self.temperature:t})
        return outputs.tolist()


    def dd_oracle_generator(self, sess, oracle, true_data_loader, fake_data_loader):
        '''
        Calculate DD distance using two generators in synthesis experiment
        See the paper for the detailed algorithm:https://arxiv.org/abs/2005.01282
        :param sess:
        :param oracle:
        :param true_data_loader:
        :param fake_data_loader:
        :return:
        '''
        all_d_true = []
        for it in range(true_data_loader.num_batch):
            y, mask = true_data_loader.next_batch()
            x = np.ones((self.batch_size, self.sequence_length), dtype=int) * 2
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length - 1]

            generator_losses = sess.run(self.eval_losses, {self.input_data: x,
                                                           self.targets: y,
                                                           self.input_mask: mask,
                                                           self.dropout_keep_place: 1.0})

            generator_losses = generator_losses * mask

            oracle_losses = sess.run(oracle.eval_losses, {oracle.input: x,
                                                          oracle.targets: y})  # [batch, sequence]
            oracle_losses = oracle_losses * mask

            generator_logp = - np.sum(generator_losses, axis=-1)  # [batch]
            oracle_logp = - np.sum(oracle_losses, axis=-1)

            t1_t = generator_logp - oracle_logp  # t1_t> 0 ==> pT1 > pT ==> z<0.5, po_sample=-1
            score = (t1_t < 0) * 1 - 1 * (t1_t > 0)
            all_d_true.append(score)
        all_d_true_score = np.sum(all_d_true) / (true_data_loader.num_batch * self.batch_size)
        print("all_d_true_score:", all_d_true_score)
        print("np.sum(all_d_true) :", np.sum(all_d_true))

        all_d_fake = []
        for it in range(true_data_loader.num_batch):
            y, mask = fake_data_loader.next_batch()
            x = np.ones((self.batch_size, self.sequence_length), dtype=int) * 2
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length - 1]

            generator_losses = sess.run(self.eval_losses, {self.input_data: x,
                                                           self.targets: y,
                                                           self.input_mask: mask,
                                                           self.dropout_keep_place: 1.0})

            generator_losses = generator_losses * mask

            oracle_losses = sess.run(oracle.eval_losses, {oracle.input: x,
                                                          oracle.targets: y})  # [batch, sequence]
            oracle_losses = oracle_losses * mask

            generator_logp = - np.sum(generator_losses, axis=-1)  # [batch]
            oracle_logp = - np.sum(oracle_losses, axis=-1)

            t1_t = generator_logp - oracle_logp  # t1_t> 0 ==> pT1 > pT ==> z<0.5, neg_sample=1
            score = (t1_t > 0) * 1 - 1 * (t1_t < 0)
            all_d_fake.append(score)
        all_d_fake_score = np.sum(all_d_fake) / (true_data_loader.num_batch * self.batch_size)
        print("all_d_fake_score:", all_d_fake_score)
        print("true_data_loader.num_batch * self.batch_size:", true_data_loader.num_batch * self.batch_size)
        print("np.sum(all_d_fake) :", np.sum(all_d_fake))

        dtt1 = (all_d_fake_score + all_d_true_score) / 2.0
        print("dtt1:", dtt1)