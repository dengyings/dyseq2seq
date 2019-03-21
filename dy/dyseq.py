import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import jieba
from recv import getinput
from output import getoutput

class dyseq(object):
    def __init__(self):
        self.init_learning_rate = 0.001
        self.input_seq_len = 5
        self.output_seq_len = 5
        self.size = 8
        self.num_decoder_symbols = 100
        self.num_encoder_symbols = 100
        self.output_dir='./model/chatbot/demo'
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2


    def word2id(self,word):
        word2id_dict = {}
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in word2id_dict:
            return word2id_dict[word]
        else:
            return None


    def get_id_list_from(self,sentence):
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            id = self.word2id(str)
            if id:
                sentence_id_list.append(id)
        return sentence_id_list


    def get_model(self,feed_previous=False):
        learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, dtype=tf.float32)
        learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
        encoder_inputs = []
        decoder_inputs = []
        target_weights = []
        for i in range(self.input_seq_len):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(self.output_seq_len + 1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        for i in range(self.output_seq_len):
            target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
        targets = [decoder_inputs[i + 1] for i in range(self.output_seq_len)]
        cell = tf.contrib.rnn.BasicLSTMCell(self.size)
        outputs, _ = seq2seq.embedding_attention_seq2seq(
                            encoder_inputs,
                            decoder_inputs[:self.output_seq_len],
                            cell,
                            num_encoder_symbols=self.num_encoder_symbols,
                            num_decoder_symbols=self.num_decoder_symbols,
                            embedding_size=self.size,
                            output_projection=None,
                            feed_previous=feed_previous,
                            dtype=tf.float32)


        loss = seq2seq.sequence_loss(outputs, targets, target_weights)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(opt.compute_gradients(loss))
        saver = tf.train.Saver(tf.global_variables())
        return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate


    def seq_to_encoder(self,input_seq):
        input_seq_array = [int(v) for v in input_seq.split()]
        encoder_input = [self.PAD_ID] * (self.input_seq_len - len(input_seq_array)) + input_seq_array
        decoder_input = [self.GO_ID] + [self.PAD_ID] * (self.output_seq_len - 1)
        encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
        decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
        target_weights = [np.array([1.0], dtype=np.float32)] * self.output_seq_len
        return encoder_inputs, decoder_inputs, target_weights


    def judge(self,text):
        word_path = './word.txt'
        word_list = [sw.replace('\n', '') for sw in open(word_path).readlines()]
        text = jieba.lcut(text)
        i = 0
        for word in text:
            if word in word_list:
                i += 1
        if i == 0:
            return True
        else:
            return False


    def predict(self):
        with tf.Session() as sess:
            encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(feed_previous=True)
            saver.restore(sess, self.output_dir)
            input_seq=getinput()
            if self.judge(input_seq):
                input_seq = input_seq.strip()
                input_id_list = self.get_id_list_from(input_seq)
                if (len(input_id_list)):
                    sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                    input_feed = {}
                    for l in range(self.input_seq_len):
                        input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                    for l in range(self.output_seq_len):
                        input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                        input_feed[target_weights[l].name] = sample_target_weights[l]
                    input_feed[decoder_inputs[self.output_seq_len].name] = np.zeros([2], dtype=np.int32)

                    outputs_seq = sess.run(outputs, input_feed)
                    outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                    if self.EOS_ID in outputs_seq:
                        outputs_seq = outputs_seq[:outputs_seq.index(self.EOS_ID)]
                    outputs_seq = [self.wordToken.id2word(v) for v in outputs_seq]
                    text = " ".join(outputs_seq)
                    return text
                else:
                    pass
            else:
                return getoutput(input_seq)


