from utils.Logger import Logger
import sys

from colorama import Fore
import tensorflow as tf

from models.lstm.LSTMD import LSTMD
from models.gpt2.GPT2D import GPT2D

from utils.config import Config
from utils.data import *

gen_models = {
    'lstm': LSTMD,
    'gpt2': GPT2D,
}

def set_gen_model(model_name):
    try:
        GenModel = gen_models[model_name.lower()]
        gen = GenModel()
        return gen
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + model_name + Fore.RESET)
        sys.exit(-2)


def fill_seq(input, padded_length, fill_token):
        input_padded = input[:]
        input_padded += [int(fill_token)] * (padded_length - len(input))
        return input_padded

def def_flags():
    flags = tf.app.flags
    flags.DEFINE_enum('gen_model', 'gpt2', list(gen_models.keys()),'Type of gen model to Training; Options=["lstm", "gpt2"]')
    flags.DEFINE_string('data', 'emnlp_news', 'Dataset for Training; Options=["emnlp_news", "oracle"]')
    flags.DEFINE_float('N', 1.0, 'N% of data used :data * N,(0<N<=1)')
    flags.DEFINE_boolean('restore', False, 'Restore pretrain models; Options=[False, True]')
    flags.DEFINE_string('experiments_name', "test", 'Experiment name for train')
    flags.DEFINE_integer('gpu', 0, 'The GPU used for training')
    return

def main(args):
    FLAGS = tf.app.flags.FLAGS
    gens = set_gen_model(FLAGS.gen_model)
    gens.FLAGS = FLAGS
    #generate num
    if FLAGS.data == 'emnlp_news':
        gens.num_generate_train = 268586
    elif FLAGS.data == 'oracle':
        gens.num_generate_train = 300000
                
    # experiment path
    experiment_path = os.path.join('experiments', FLAGS.experiments_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    print(f"{Fore.BLUE}Experiment path: {experiment_path}{Fore.RESET}")

    # tempfile
    tmp_path = os.path.join(experiment_path, 'tmp')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    #generate training set for discriminator training
    gens.generator_file_pkl = os.path.join(tmp_path, 'generator.pkl')
    #generate  test set for relevant evaluation
    gens.generator_test_file = os.path.join(tmp_path, 'generator_test.txt')
    #generate valid set and input it into the discriminator
    #together with the real valid set for observation
    #of the discriminator training process
    gens.generator_valid_file = os.path.join(tmp_path, 'generator_valid.txt')
    #Text content of generator_test.txt
    gens.text_file = os.path.join(tmp_path, 'text_file.txt')#words

    ####
    #The True samples in the synthesis experiment are the training set, test set and validation set respectively
    ####
    gens.oracle_generator_file_pkl= os.path.join(tmp_path, 'oracle_generator.pkl')
    gens.oracle_generator_test_file = os.path.join(tmp_path, 'oracle_generator_test.txt')
    gens.oracle_generator_valid_file = os.path.join(tmp_path, 'oracle_generator_valid.txt')
    # Text content of oracle_generator_test.txt
    gens.oracle_text_file = os.path.join(tmp_path, 'oracle_text_file.txt') #words

    # Log file
    gens.log = os.path.join(
        experiment_path, f'log-{FLAGS.experiments_name}.csv')
    if os.path.exists(gens.log) and not FLAGS.restore:
        print(f"{Fore.RED}[Error], Log file exist!{Fore.RESET}")
        exit(-3)

    # Config file
    config_file = os.path.join(experiment_path, 'config.json')
    if not os.path.exists(config_file):
        config_file = os.path.join('models', FLAGS.gen_model, 'config.json')
        # copy config file
        from shutil import copyfile
        copyfile(config_file, os.path.join(experiment_path, 'config.json'))
        if not os.path.exists(config_file):
            print(f"{Fore.RED}[Error], Config file not exist!{Fore.RESET}")
    print(f"{Fore.BLUE}Using config: {config_file}{Fore.RESET}")
    config = Config(config_file)
    gens.set_config(config)

    # output path
    gens.output_path = os.path.join(experiment_path, 'output')
    if not os.path.exists(gens.output_path):
        os.mkdir(gens.output_path)

    # save path
    gens.save_path = os.path.join(experiment_path, 'ckpts')
    gens.restore = FLAGS.restore
    if not os.path.exists(gens.save_path):
        os.mkdir(gens.save_path)

    # summary path
    gens.summary_path = os.path.join(experiment_path, 'summary')
    if not os.path.exists(gens.summary_path):
        os.mkdir(gens.summary_path)

    # print log
    path = os.path.join(experiment_path, f'log-print-{FLAGS.experiments_name}.txt')
    sys.stdout = Logger(path)
    if not os.path.exists(path):
        print(f"{Fore.RED}[Error], print_log file not exist!{Fore.RESET}")
        exit(-3)

    # preprocess real data
    data_file = f"data/emnlp_news.txt"
    valid_data_file = f"data/validdata/valid_emnlp_news.txt"
    test_data_file = f"data/testdata/test_emnlp_news.txt"
    vocab_file = f"data/vocab/emnlp_news_word_dict.vocab.pkl"
    train_ids_file = f"data/vocab/emnlp_news_train_ids.pkl"
    valid_ids_file = f"data/vocab/emnlp_news_valid_ids.pkl"
    test_ids_file = f"data/vocab/emnlp_news_test_ids.pkl"

    # dataset creation
    dataset_train, word_dict = tokenize(data_file=data_file, \
            data_ids_file=train_ids_file, vocab_file=vocab_file, train=True)
    gens.sequence_length = word_dict.max_seq_len
    gens.vocab_size = word_dict.__len__()
    gens.wi_dict = word_dict.word2idx
    gens.iw_dict = word_dict.idx2word
    if FLAGS.data == 'emnlp_news':
        dataset_test,  word_dict = tokenize(data_file=test_data_file, \
                data_ids_file=test_ids_file, vocab_file=vocab_file, train=False, word_dict=word_dict)
        dataset_valid,  word_dict = tokenize(data_file=valid_data_file, \
                data_ids_file=valid_ids_file, vocab_file=vocab_file, train=False, word_dict=word_dict)
        gens.test_code = [fill_seq(sentence, padded_length=gens.sequence_length, fill_token=0) for sentence in dataset_test]
        gens.train_code = [fill_seq(sentence, padded_length=gens.sequence_length, fill_token=0) for sentence in dataset_train]
        gens.valid_code = [fill_seq(sentence, padded_length=gens.sequence_length, fill_token=0) for sentence in dataset_valid]
        #train
        gens.train_real()
    elif FLAGS.data == 'oracle':
        gens.train_oracle()
    else:
        print("data set error")

    '''
    get lm_vs_rlm score
    '''
    # gens.lm_vs_rlm(model_path)

    '''
    bleu_vs_sbleu
    real_text and fake_text are words, not ids.
    '''
    # gens.bleu_vs_sbleu(real_text, fake_text)

    '''
    fed
    real_text and fake_text are words, not ids.
    require Tensorflow 2.0
    '''
    # gens.fed(real_text, fake_text)


if __name__ == '__main__':
    def_flags()
    tf.app.run()
