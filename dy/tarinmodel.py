import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import jieba
import random


input_seq_len = 5
output_seq_len = 5
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
size = 8
init_learning_rate = 0.001
min_freq = 10
class WordToken(object):
    def __init__(self):
        # 最小起始id号, 保留的用于表示特殊标记
        self.START_ID = 4
        self.word2id_dict = {}
        self.id2word_dict = {}


    def load_file_list(self, file_list, min_freq):
        """
        加载样本文件列表，全部切词后统计词频，按词频由高到低排序后顺次编号
        并存到self.word2id_dict和self.id2word_dict中
        """
        words_count = {}
        for file in file_list:
            with open(file, 'r') as file_object:
                for line in file_object.readlines():
                    line = line.strip()
                    seg_list = jieba.cut(line)
                    for str in seg_list:
                        if str in words_count:
                            words_count[str] = words_count[str] + 1
                        else:
                            words_count[str] = 1

        sorted_list = [[v[1], v[0]] for v in words_count.items()]
        sorted_list.sort(reverse=True)
        for index, item in enumerate(sorted_list):
            word = item[1]
            if item[0] < min_freq:
                break
            self.word2id_dict[word] = self.START_ID + index
            self.id2word_dict[self.START_ID + index] = word
        return index

    def word2id(self, word):
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return None

    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return None
wordToken = WordToken()


max_token_id = wordToken.load_file_list(['./data/question.txt', './data/answer.txt'],10)
num_encoder_symbols = max_token_id + 5
num_decoder_symbols = max_token_id + 5


output_dir='./model/data/demo'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_model(feed_previous=False):
    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]
    cell = tf.contrib.rnn.BasicLSTMCell(size)
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update = opt.apply_gradients(opt.compute_gradients(loss))
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate


def get_samples(train_set, batch_num):
    raw_encoder_input = []
    raw_decoder_input = []
    if batch_num >= len(train_set):
        batch_train_set = train_set
    else:
        random_start = random.randint(0, len(train_set)-batch_num)
        batch_train_set = train_set[random_start:random_start+batch_num]
    for sample in batch_train_set:
        raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
        raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
    for length_idx in range(output_seq_len):
        decoder_inputs.append(np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
        target_weights.append(np.array([
            0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in raw_decoder_input
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights


def get_id_list_from(sentence):
    sentence_id_list = []
    seg_list = jieba.cut(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(id)
    return sentence_id_list


def get_train_set():
    global num_encoder_symbols, num_decoder_symbols
    train_set = []
    with open('./data/question.txt', 'r') as question_file:
        with open('./data/answer.txt', 'r') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    question = question.strip()
                    answer = answer.strip()

                    question_id_list = get_id_list_from(question)
                    answer_id_list = get_id_list_from(answer)
                    if len(question_id_list) > 0 and len(answer_id_list) > 0:
                        answer_id_list.append(EOS_ID)
                        train_set.append([question_id_list, answer_id_list])
                else:
                    break
    print(train_set)
    return train_set



def train():
    train_set = get_train_set()
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model()
        sess.run(tf.global_variables_initializer())
        previous_losses = []
        for step in range(10000):
            sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples(train_set, 1000)
            input_feed = {}
            for l in range(input_seq_len):
                input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(output_seq_len):
                input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[target_weights[l].name] = sample_target_weights[l]
            input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])], dtype=np.int32)
            [loss_ret, _] = sess.run([loss, update], input_feed)
            if step % 100 == 0:
                print( 'step=', step, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())
                if len(previous_losses) > 5 and loss_ret > max(previous_losses[-5:]):
                    sess.run(learning_rate_decay_op)
                previous_losses.append(loss_ret)
                saver.save(sess, output_dir)



if __name__ == "__main__":
    tf.reset_default_graph()
    train()