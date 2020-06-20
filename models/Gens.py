from abc import abstractmethod, ABCMeta

from utils.utils import init_sess
from utils.data import *
import os
import numpy as np
import tensorflow as tf
import pickle
from utils.metrics.Nll import Nll



class Gens(metaclass=ABCMeta):

    def __init__(self):
        self.FLAGS = None
        #model
        self.oracle = None
        self.generator = None
        self.re_generator = None
        self.discriminator = None
        #data load
        self.train_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None
        self.fake_data_loader = None
        self.dis_train_data_loader = None
        self.dis_valid_data_loader = None
        # temp file
        self.generator_file_pkl = None
        self.generator_test_file = None
        self.generator_valid_file = None
        self.text_file = None
        self.oracle_generator_file_pkl = None
        self.oracle_generator_test_file = None
        self.oracle_generator_valid_file = None
        self.oracle_text_file = None
        # pathes
        self.output_path = None
        self.save_path = None
        self.summary_path = None
        # dict
        self.wi_dict = None
        self.iw_dict = None
        #common
        self.sequence_length = None
        self.vocab_size = None
        self.sess = init_sess()
        self.metrics = list()
        self.log = None
        self.epoch = 0
        #generate num
        self.num_generate_train = None
        #train pkl file
        self.train_code = None
        self.valid_code = None
        self.test_code = None

    def set_config(self, config):
        self.__dict__.update(config.dict)

    def set_generator(self, generator=None, re_generator=None, oracle=None):
        self.generator = generator
        self.re_generator = re_generator
        self.oracle=oracle

    def set_discriminator(self, discriminator=None):
        self.discriminator = discriminator

    def set_data_loader(self, train_loader, valid_loader, test_loader, fake_loader, dis_train_loader, dis_valid_loader):
        #true sample data loader
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.test_data_loader = test_loader
        #fake sample data loader
        self.fake_data_loader = fake_loader
        #dis data lodaer
        self.dis_train_data_loader = dis_train_loader
        self.dis_valid_data_loader = dis_valid_loader

    def set_sess(self, sess):
        self.sess = sess

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # in use
        self.epoch = 0
        return

    def add_metric(self, metric):
        self.metrics.append(metric)

    def evaluate_scores(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print(f"time elapsed of {metric.get_name()}: {toc - tic:.1f}s")
            print(metric.get_name() + ":" + str(score))
            scores.append(score)
        print(log)
        return scores

    def evaluate(self):
        head = ["epoch"]
        for metric in self.metrics:
            head.append(metric.get_name())
        with open(self.log, 'a') as log:
            if self.epoch == 0 or self.epoch == 1:
                log.write(','.join(head) + '\n')
            scores = self.evaluate_scores()
            log.write(','.join([str(s) for s in scores]) + '\n')
        return dict(zip(head, scores))

    def get_real_test_file(self, temperature=1.0, type=None):
        '''
         Generate Samples test id to word
        :return:
        '''
        if type == "oracle":
            generator_test = self.oracle_generator_test_file
            text_file = self.oracle_text_file
        else:
            generator_test = self.generator_test_file
            text_file = self.text_file

        with open(generator_test, 'r') as file:
            codes = get_tokenlized(generator_test)
        output = id_to_words(ids=codes, idx2word=self.iw_dict)
        with open(text_file, 'w', encoding='utf-8') as outfile:
            outfile.write(output)
        output_file = os.path.join(self.output_path, f"data{int(self.FLAGS.N*100)}_t{int(temperature * 100)}.txt")
        with open(output_file, 'w', encoding='utf-8') as of:
            of.write(output)

    def generate_samples(self, temperature=1.0, generator=None, type=None):
        '''
        Three samples are generated for discriminator training, verification and testing.
        Stored in three files.
        Samples train num: num_generate_train
        Samples test num: 10000
        Samples valid num: 10000
        :param temperature:When the generator is LSTM, temperature here only works；
                            gpt-2's temperature is set when the model is initialized
        :param generator:
        :param type:
        :return:
        '''

        if type == "oracle":
            generator_pkl = self.oracle_generator_file_pkl
            generator_valid = self.oracle_generator_valid_file
            generator_test = self.oracle_generator_test_file
        else:
            generator_pkl = self.generator_file_pkl
            generator_valid = self.generator_valid_file
            generator_test = self.generator_test_file

        # Generate Samples as fake train set
        generated_samples = []
        for _ in range(int(self.num_generate_train / self.batch_size)+1):
            generated_samples.extend(generator.generate(self.sess, temperature))
        with open(generator_pkl, 'wb') as out:
            pickle.dump(generated_samples[:self.num_generate_train], out)
        
        # Generate Samples as fake valid set
        generated_samples_valid = []
        for _ in range(int(10000 / self.batch_size) + 1):
            generated_samples_valid.extend(generator.generate(self.sess, temperature))
        with open(generator_valid, 'w') as fout:
            for sent in generated_samples_valid[:10000]:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)
    
        # Generate Samples as fake test set
        generated_samples_test = []
        for _ in range(int(10000 / self.batch_size)+1):
            generated_samples_test.extend(generator.generate(self.sess, temperature))
        with open(generator_test, 'w') as fout:
            for sent in generated_samples_test[:10000]:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)

        return generated_samples[:self.num_generate_train], generated_samples_valid[:10000], generated_samples_test[:10000]

    def init_real_metric(self):

        from utils.metrics.Nll import Nll
        from utils.metrics.PPL import PPL
        #from utils.metrics.FED import FED #FED requires tensorflow2.0
        from utils.metrics.Bleu import Bleu
        from utils.metrics.SelfBleu import SelfBleu

        if self.valid_nll:
            valid_nll = Nll(self.valid_data_loader, self.generator, self.sess)
            valid_nll.set_name('valid_nll')
            self.add_metric(valid_nll)
        if self.selfbleu:
            for i in range(2,6):
                selfbleu = SelfBleu(test_text=self.text_file, gram=i)
                selfbleu.set_name(f"Selfbleu{i}")
                self.add_metric(selfbleu)
        if self.bleu:
            dataset = self.FLAGS.data
            if dataset == "emnlp_news":
                real_text = 'data/testdata/test_emnlp_news.txt'
            else:
                raise ValueError
            for i in range(2,6):
                bleu = Bleu(
                    test_text=self.text_file,
                    real_text=real_text, gram=i)
                bleu.set_name(f"Bleu{i}")
                self.add_metric(bleu)

    def save_summary(self):
        # summary writer
        self.sum_writer = tf.summary.FileWriter(
            self.summary_path, self.sess.graph)
        return self.sum_writer


    def get_distance(self, fake_file, discriminator, datatype, true_data_loader, epoch, writer):
        '''
        Calculate the dd distance, accuracy, loss, etc. of the discriminator
        :param fake_file:
        :param discriminator:
        :param datatype:
        :param true_data_loader:
        :param epoch:
        :param writer:
        :return:
        '''
        if isinstance(fake_file, list):
            self.fake_data_loader.create_batches_train_list(fake_file)
        else:
            self.fake_data_loader.create_batches(fake_file)

        true_correct_all = []
        true_error_all = []
        true_loss_all = []
        true_sum_all = []
        for _ in range(true_data_loader.num_batch):
            x_batch_t, _ = true_data_loader.next_batch()
            y_batch_t = [[0, 1] for _ in range(self.batch_size)]
            feed_t = {
                discriminator.input_x: x_batch_t,
                discriminator.input_y: y_batch_t,
                discriminator.dropout_keep_prob: 1.0
            }
            true_sum, true_loss, true_correct, true_error = self.sess.run(
                [discriminator.true_sum, discriminator.loss, discriminator.true_correct, discriminator.true_error],
                feed_t)
            true_correct_all.append(np.sum(true_correct))
            true_error_all.append(np.sum(true_error))
            true_loss_all.append(true_loss)
            true_sum_all.append(true_sum)

        fake_correct_all = []
        fake_error_all = []
        fake_loss_all = []
        fake_sum_all = []
        for _ in range(true_data_loader.num_batch):
            x_batch_f, _ = self.fake_data_loader.next_batch()
            y_batch_f = [[1, 0] for _ in range(self.batch_size)]
            feed = {
                discriminator.input_x: x_batch_f,
                discriminator.input_y: y_batch_f,
                discriminator.dropout_keep_prob: 1.0
            }
            fake_sum, fake_loss, fake_correct, fake_error = self.sess.run(
                [discriminator.fake_sum, discriminator.loss, discriminator.fake_correct, discriminator.fake_error],
                feed)
            fake_correct_all.append(np.sum(fake_correct))
            fake_error_all.append(fake_error)
            fake_loss_all.append(fake_loss)
            fake_sum_all.append(fake_sum)

        loss = (np.sum(true_loss_all) + np.sum(fake_loss_all)) / (true_data_loader.num_batch * self.batch_size * 2)
        accuracy = (np.sum(true_correct_all) + np.sum(fake_correct_all)) / (
                    2 * true_data_loader.num_batch * self.batch_size)
        dd = (np.sum(true_correct_all) + np.sum(fake_correct_all) - np.sum(true_error_all) - np.sum(fake_error_all)) / (
                    2 * true_data_loader.num_batch * self.batch_size)
        print(f"accuracy {datatype}:", str(accuracy))
        print(f"loss {datatype}：", str(loss))
        print(f"dd {datatype}：", str(dd))
        true_score_avr = np.sum(true_sum_all) / (true_data_loader.num_batch * self.batch_size)
        fake_score_avr = np.sum(fake_sum_all) / (true_data_loader.num_batch * self.batch_size)

        #tensoboard info
        if "valid" in datatype :
            feed = {
                discriminator.valid_acc: accuracy,
                discriminator.valid_dd: dd,
                discriminator.valid_loss: loss,
                discriminator.valid_t_avr: true_score_avr,
                discriminator.valid_f_avr: fake_score_avr
            }
            merge_valid = self.sess.run(discriminator.merge_valid,
                feed)
            writer.add_summary(merge_valid, epoch)
        elif "test" in datatype:
            feed = {
                discriminator.test_acc: accuracy,
                discriminator.test_dd: dd,
                discriminator.test_loss: loss,
                discriminator.test_t_avr: true_score_avr,
                discriminator.test_f_avr: fake_score_avr
                }
            merge_test = self.sess.run(discriminator.merge_test,
                feed)
            writer.add_summary(merge_test, epoch)

        return accuracy, dd, loss


    def lm_scores(self, generator, fake_data_loader, sess):
        lm = Nll(fake_data_loader, generator, sess)
        lm_score = lm.get_score()
        print("lm_score:", lm_score)
        return lm_score

    def train_real(self):
        pass


class Gen(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def run_epoch(self):
        pass


class Dis(metaclass=ABCMeta):

    def __init__(self):
        pass

