from time import time

from models.Gens import Gens
from models.lstm.DataLoader import DataLoader,BalanceDisDataloader
from models.lstm.Discriminator import Discriminator
from models.lstm.LSTM import Generator
from utils.oracle.OracleGPT2 import OracleGPT2
from utils.utils import *
from colorama import Fore
import tensorflow.contrib.slim as slim
import pickle, math
import numpy as np
from utils.metrics.Nll import Nll
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu
from utils.metrics.FED import FED


class LSTMD(Gens):
    def __init__(self):
        super().__init__()


    def init_oracle_trainng(self):
        '''
        Synthetic experiment related model initialization
        :return:
        '''
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope(f"generator_gpt2", reuse=None, initializer=initializer):
            oracle = OracleGPT2(num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                                sequence_length=self.sequence_length, \
                                generator_name=f"generator_gpt2", start_token=self.start_token, \
                                learning_rate=self.gen_lr, temperature=self.temperature, n_embd=self.n_embd,
                                n_head=self.n_head, n_layer=self.n_layer, dropout_keep_prob=self.g_dropout_keep_prob)

        with tf.variable_scope(f"oracle_generator_lstm", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                                     hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                     generator_name=f"oracle_generator_lstm",
                                     learning_rate=self.gen_lr, grad_clip=self.grad_clip,
                                     dropout_keep_prob=self.g_dropout_keep_prob, start_token=self.start_token,
                                     num_layers_gen=self.num_layers_gen)

        self.set_generator(generator=generator, oracle=oracle)

        with tf.variable_scope(f"oracle_discriminator_cnn"):
            discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                          vocab_size=self.vocab_size,
                                          discriminator_name=f'oracle_discriminator_cnn',
                                          emd_dim=self.emb_dim, filter_sizes=self.filter_size,
                                          num_filters=self.num_filters,l2_reg_lambda=self.l2_reg_lambda, dis_lr=self.dis_lr)

        self.set_discriminator(discriminator=discriminator)
        #dataloder
        train_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        #discriminator　dataloder
        dis_train_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length, padding_token=0)
        dis_valid_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length, padding_token=0)

        self.set_data_loader(train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader, fake_loader=fake_dataloader, dis_train_loader=dis_train_dataloader, dis_valid_loader=dis_valid_dataloader)

    def train_discriminator(self, discriminator):
        '''
        Discriminator training
        :param discriminator:
        :return:
        '''
        for step in range(self.dis_train_data_loader.num_batch_po):
            x_batch, y_batch = self.dis_train_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: self.d_dropout_keep_prob
            }
            self.sess.run(discriminator.train_op, feed)

    def init_real_trainng(self):
        '''
        Real data experiment  related model initialization
        :return:
        '''
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope(f"generator_lstm", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                generator_name=f"generator_lstm", learning_rate=self.gen_lr, grad_clip=self.grad_clip,
                                dropout_keep_prob=self.g_dropout_keep_prob, start_token=self.start_token, num_layers_gen=self.num_layers_gen)

        self.set_generator(generator=generator)
 
        with tf.variable_scope(f"discriminator_cnn"):
            discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                          vocab_size=self.vocab_size,
                                          discriminator_name=f'discriminator_cnn',
                                          emd_dim=self.emb_dim, filter_sizes=self.filter_size,
                                          num_filters=self.num_filters,
                                          l2_reg_lambda=self.l2_reg_lambda, dis_lr=self.dis_lr)

        self.set_discriminator(discriminator=discriminator)
      
        #dataloder
        train_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        #discriminator　dataloder
        dis_train_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length, padding_token=0)
        dis_valid_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length, padding_token=0)

        self.set_data_loader(train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader, fake_loader=fake_dataloader, dis_train_loader=dis_train_dataloader, dis_valid_loader=dis_valid_dataloader)

    def train_real(self):
        '''
        Real data experiment
        :return:
        '''
        self.init_real_trainng()
        ###
        #init_metric:NLL
        ###
        gen_valid_nll = Nll(self.valid_data_loader, self.generator, self.sess)


        self.valid_data_loader.create_batches_train_list(self.valid_code)
        self.train_data_loader.create_batches_train_list(np.array(self.train_code)[:math.ceil(self.num_generate_train * self.FLAGS.N)].tolist())
        self.test_data_loader.create_batches_train_list(self.test_code)
        self.sess.run(tf.global_variables_initializer())

        saver_variables = slim.get_variables_to_restore(include=[f"generator_lstm"])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)

        # summary writer
        self.writer = self.save_summary()

        if self.restore:
            restore_from = tf.train.latest_checkpoint(self.save_path)
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
        else:
            best_nll = 1000
            print('start train generator:')
            for epoch in range(self.train_gen_num):
                start = time()
                loss = self.generator.run_epoch(self.sess, self.train_data_loader, self.writer, epoch)
                end = time()
                print('epoch:' + str(epoch) + ' loss: ' + str(loss) + ' \t time:' + str(end - start))
                if (epoch + 1) % self.ntest == 0:
                    values = gen_valid_nll.get_score()
                    if values < best_nll:
                        best_nll = values
                        # save pre_train
                        saver.save(self.sess,
                                   os.path.join(self.save_path, f'data{int(self.FLAGS.N * 100)}_train_gen_best'))
                        print('gen store')
                self.add_epoch()
            restore_from = os.path.join(self.save_path, f'data{int(self.FLAGS.N * 100)}_train_gen_best')
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")

            print('start train discriminator:')
            saver_variables = slim.get_variables_to_restore(include=["discriminator_cnn"])
            saver = tf.train.Saver(saver_variables, max_to_keep=20)
            self.generate_samples(self.temperature, self.generator)
            self.get_real_test_file(self.temperature)
            with open(self.generator_file_pkl, 'rb')  as inf:
                self.generator_code = pickle.load(inf)
            self.dis_train_data_loader.load_train_data_list(self.train_code, self.generator_code)
            self.dis_valid_data_loader.load_train_data_list_file(self.valid_code, self.generator_valid_file)
            self.train_data_loader.create_batches_train_list(self.train_code)
            acc_valid_best = 0
            acc_test_best = 0
            for epoch in range(self.train_dis_num):
                print(f"{int(self.FLAGS.N*100)}d epoch:" + str(epoch))
                self.train_discriminator(self.discriminator)

                accuracy_valid, dd_valid, loss_valid = self.get_distance(self.generator_valid_file, self.discriminator, \
                                                                         f"{int(self.FLAGS.N * 100)}d_valid",
                                                                         self.valid_data_loader, epoch,
                                                                         self.writer)
                if accuracy_valid > acc_valid_best:
                        acc_valid_best = accuracy_valid
                        saver.save(self.sess, os.path.join(self.save_path, f'train_dis'))
            print("acc_valid_best:", acc_valid_best)
            restore_from = os.path.join(self.save_path, "train_dis")
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            accuracy_test, dd_test, loss_test = self.get_distance(self.generator_test_file, self.discriminator, \
                                                                  f"{int(self.FLAGS.N*100)}d_test", self.test_data_loader, epoch,
                                                                  self.writer)
            print("acc_test:", accuracy_test)

    def train_oracle(self):
        self.init_oracle_trainng()
        ###
        #init_metric:NLL
        ###
        gen_valid_nll = Nll(self.valid_data_loader, self.generator, self.sess)
        self.sess.run(tf.global_variables_initializer())

        #Load oracle model
        saver_variables = slim.get_variables_to_restore(include=[f"generator_gpt2"])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)
        restore_from = tf.train.latest_checkpoint(os.path.join(self.save_path, "oracle"))
        saver.restore(self.sess, restore_from)
        print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")

        #The oracle model generates True samples of synthetic experiments: oracle_generator.pkl, oracle_generator_test.txt, oracle_generator_valid.txt
        self.generate_samples(temperature=self.temperature, generator=self.oracle, type="oracle")
        self.get_real_test_file(temperature=self.temperature, type="oracle")
        with open(self.oracle_generator_file_pkl, 'rb')  as inf:
            self.train_code = pickle.load(inf)
        self.train_data_loader.create_batches_train_list(np.array(self.train_code)[:math.ceil(self.num_generate_train * self.FLAGS.N)].tolist())
        self.valid_data_loader.create_batches(self.oracle_generator_valid_file)
        self.test_data_loader.create_batches(self.oracle_generator_test_file)

        saver_variables = slim.get_variables_to_restore(include=[f"oracle_generator_lstm"])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)

        # summary writer
        self.writer = self.save_summary()

        if self.restore:
            restore_from = tf.train.latest_checkpoint(self.save_path)
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
        else:
            best_nll = 1000
            print('start train generator:')
            for epoch in range(self.train_gen_num):
                start = time()
                loss = self.generator.run_epoch(self.sess, self.train_data_loader, self.writer, epoch)
                end = time()
                print('epoch:' + str(epoch) + ' loss: ' + str(loss) + ' \t time:' + str(end - start))
                if (epoch + 1) % self.ntest == 0:
                    values = gen_valid_nll.get_score()
                    if values < best_nll:
                        best_nll = values
                        # save pre_train
                        saver.save(self.sess, os.path.join(self.save_path, f'oracle_data{int(self.FLAGS.N*100)}_train_gen_best'))
                        print('gen store')
                self.add_epoch()
            restore_from = os.path.join(self.save_path, f'oracle_data{int(self.FLAGS.N*100)}_train_gen_best')
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")

            self.generate_samples(self.temperature, self.generator)
            self.get_real_test_file(self.temperature)
            self.fake_data_loader.create_batches(self.generator_test_file)
            ####
            #Calculate DD distance using two generators in synthesis experiment
            ####
            self.generator.dd_oracle_generator(self.sess, self.oracle, self.test_data_loader, self.fake_data_loader)

            print('start train discriminator:')
            with open(self.generator_file_pkl, 'rb')  as inf:
                self.generator_code = pickle.load(inf)
            self.dis_train_data_loader.load_train_data_list(self.train_code, self.generator_code)
            self.dis_valid_data_loader.load_train_data(self.oracle_generator_valid_file, self.generator_valid_file)
            self.train_data_loader.create_batches_train_list(self.train_code)
            acc_valid_best = 0
            for epoch in range(self.train_dis_num):
                print(f"{int(self.FLAGS.N*100)}d epoch:" + str(epoch))
                self.train_discriminator(self.discriminator)


                accuracy_valid, dd_valid, loss_valid = self.get_distance(self.generator_valid_file, self.discriminator, \
                                                                         f"{int(self.FLAGS.N * 100)}d_valid",
                                                                         self.valid_data_loader, epoch,
                                                                         self.writer)
                if accuracy_valid > acc_valid_best:
                        acc_valid_best = accuracy_valid
                        saver.save(self.sess, os.path.join(self.save_path, 'oracle_train_dis'))
            print("acc_valid_best:", acc_valid_best)
            restore_from = os.path.join(self.save_path, "oracle_train_dis")
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            accuracy_test, dd_test, loss_test = self.get_distance(self.generator_test_file, self.discriminator, \
                                                                  f"{int(self.FLAGS.N*100)}d_test", self.test_data_loader, epoch,
                                                                  self.writer)
            print("acc_test:", accuracy_test)


    ###
    #Evaluation: lm_vs_rlm, bleu_vs_sbleu, fed
    ###

    def rlm_scores(self, re_generator, fake_data_loader, sess, test_data_loader, valid_data_loader, writer):
        test_rlm = Nll(test_data_loader, re_generator, sess)
        valid_rlm = Nll(valid_data_loader, re_generator, sess)
        print('start train re-generator:')
        valid_rlm_best = 1000
        test_rlm_best = 0
        self.re_gen_num = 80
        for epoch in range(self.re_gen_num):
            start = time()
            loss = re_generator.run_epoch(sess, fake_data_loader, writer, epoch)
            end = time()
            print('epoch:' + str(epoch) + ' loss: ' + str(loss) + ' \t time:' + str(end - start))

            test_rlm_score = test_rlm.get_score()
            valid_rlm_score = valid_rlm.get_score()
            print('valid_rlm_score:' + str(valid_rlm_score) + '   test_rlm_score: ' + str(test_rlm_score))
            if (epoch + 1) % self.ntest == 0:
                if valid_rlm_score < valid_rlm_best:
                    valid_rlm_best = valid_rlm_score
                    test_rlm_best = test_rlm_score
        print("*"*50)
        print('valid rlm best:' + str(valid_rlm_best) + 'test rlm best: ' + str(test_rlm_best))

        return valid_rlm_best, test_rlm_best

    def lm_vs_rlm(self, model_path):
        '''
        Run the train_real method and get the model, fake_train_code, fake_test_code
        :param model_path:
        :param fake_train_code_file: .pkl file
        :param fake_test_code_file: ___test.txt file
        :return:
        '''

        # dataloder
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)

        self.set_data_loader(train_loader=None, valid_loader=valid_dataloader, test_loader=test_dataloader,
                             fake_loader=fake_dataloader, dis_train_loader=None,
                             dis_valid_loader=None)

        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        if self.FLAGS.data == "oracle":
            self.valid_data_loader.create_batches(self.oracle_generator_valid_file)
            self.test_data_loader.create_batches(self.oracle_generator_test_file)
            name = "oracle_generator_lstm"
        else:
            self.valid_data_loader.create_batches_train_list(self.valid_code)
            self.test_data_loader.create_batches_train_list(self.test_code)
            name = "generator_lstm"
        with tf.variable_scope(name, reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                                     hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                     generator_name=name,
                                     learning_rate=self.gen_lr, grad_clip=self.grad_clip,
                                     dropout_keep_prob=self.g_dropout_keep_prob, start_token=self.start_token,
                                     num_layers_gen=self.num_layers_gen)
        self.set_generator(generator=generator)
        self.sess.run(tf.global_variables_initializer())


        # ++ Saver
        # saver_variables = tf.global_variables
        saver_variables = slim.get_variables_to_restore(include=[name])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)
        # ++ ====================

        # summary writer
        self.writer = self.save_summary()

        print("-- lstm lm_vs_rlm--")
        saver.restore(self.sess, model_path)
        print(f"{Fore.BLUE}Restore from : {model_path}{Fore.RESET}")

        #Generate samples and load
        self.generate_samples(self.temperature, self.generator)
        self.fake_data_loader.create_batches(self.generator_test_file)
        self.lm_scores(self.generator, self.fake_data_loader, self.sess)
        with open(self.generator_file_pkl, 'rb') as inf:
            fake_train_code = pickle.load(inf)
        self.fake_data_loader.create_batches_train_list(fake_train_code)

        with tf.variable_scope("re_generator", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                                     hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                     generator_name="re_generator",
                                     learning_rate=self.gen_lr, grad_clip=self.grad_clip,
                                     dropout_keep_prob=self.g_dropout_keep_prob, start_token=self.start_token,
                                     num_layers_gen=self.num_layers_gen)

        self.set_generator(generator=generator)
        self.sess.run(tf.global_variables_initializer())

        self.rlm_scores(self.generator, self.fake_data_loader, self.sess, self.test_data_loader, self.valid_data_loader,
                        self.writer)

    def bleu_vs_sbleu(self, real_text, fake_text):
        bleu = Bleu(test_text=fake_text, real_text=real_text, gram=5)
        sbleu = SelfBleu(test_text=fake_text, gram=5)
        print("Bleu:", bleu.get_score(), "SelfBleu", sbleu.get_score())

    def fed(self, real_text, fake_text):
        '''
        require Tensorflow 2.0
        and hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
        :param real_text:
        :param fake_text:
        :return:
        '''
        fed = FED(test_text=fake_text, real_text=real_text)
        print("FED:", fed.get_score())