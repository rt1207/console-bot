#!/usr/bin/env python
#-*- coding:utf-8 -*-

import time
from slack_model import SlackModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from EncoderDecoderModelAttention import EncoderDecoderModelAttention
from EncoderDecoderModelForward import EncoderDecoderModelForward
from Attention.attention_dialogue import AttentionDialogue
from chainer import serializers
from util.vocabulary import Vocabulary
import MeCab
from util.XP import XP

import six
import numpy
import re

class SlackApp():
    """
    Slack Call app
    You preapre the chainer model, You execute the bellow command, you can play the dialogue app
    Example
        python app.py
    """

    def __init__(self, data_model):
        """
        Iniital Setting
        :param data_model: Setting Slack Model. Slack Model has the a lot of paramater
        """
        # self.slack_channel = data_model.slack_channel
        self.data = ""
        self.parameter = data_model.parameter_dict
        self.model_name = "ChainerDialogue"
        self.generation_limit = 200
        """
        We confirm channel number
        https://api.slack.com/methods/channels.list
        """
        self.chan = data_model.chan
        self.usr = data_model.user
        self.mecab_dict = data_model.mecab_dict
        self.Mecab = MeCab.Tagger("-Owakati")
        XP.set_library(False, 0)
        self.XP = XP

    def call_method(self):
        if True: #len(self.data) >= 1 and "text" in self.data[0]:
            #print(self.data[0]["text"])
            if True: # "chainer:" in self.data[0]["text"]:
                # input sentence
                src_batch = self.__input_sentence()
                print(src_batch)
                # predict
                hyp_batch = self.__predict_sentence(src_batch)
                print(hyp_batch)
                # show predict word
                word = ''.join(hyp_batch[0]).replace("</s>", "")
                print(word)
                # self.slack_channel.api_call("chat.postMessage", user=self.usr, channel=self.chan, text=word))
            if "chainer_train" in self.data[0]["text"]:
                self.__setting_parameter()
                model = AttentionDialogue.load_spec(self.model_name + '.spec', self.XP)
                dialogue = EncoderDecoderModelAttention(self.parameter)
                serializers.load_hdf5(self.model_name + '.weights', model)
                dialogue.attention_dialogue = model
                dialogue.word2vecFlag = False
                dialogue.train()

    def __input_sentence(self):
        """
        return sentence for chainer predict
        """
        text = self.__mecab_method('test: ユーザーの発話がはいります')
        data = [text]
        src_batch = [x + ["</s>"] * (self.generation_limit - len(x) + 1) for x in data]
        return src_batch

    def __predict_sentence(self, src_batch):
        """
        predict sentence
        :param src_batch: get the source sentence
        :return:
        """
        dialogue = EncoderDecoderModelAttention(self.parameter)
        src_vocab = Vocabulary.load(self.model_name + '.srcvocab')
        trg_vocab = Vocabulary.load(self.model_name + '.trgvocab')
        model = AttentionDialogue.load_spec(self.model_name + '.spec', self.XP)
        # FIXME: serializers.load_hdf5(self.model_name + '.weights', model)
        hyp_batch = dialogue.forward_implement(src_batch, None, src_vocab, trg_vocab, model, False, self.generation_limit)
        return hyp_batch

    def __setting_parameter(self):
        """
        setteing each patamater
        """
        self.parameter["word2vec"] = self.model_name
        # train_path = "../twitter/"
        # self.parameter["source"] = train_path + "source_twitter_data.txt"
        # self.parameter["target"] = train_path + "replay_twitter_data.txt"

    def __mecab_method(self, text):
        """
        Call the mecab method
        :param text: user input text
        :return:
        """
        mecab_text = self.Mecab.parse(text)
        return mecab_text.split(" ")

if __name__ == '__main__':
    # data_model = SlackModel()
    # data_model.read_config()
    # slack = SlackApp(data_model)
    # slack.call_method()

    n_result = 5  # number of search result to show

    with open('word2vec/word2vec.model', 'r') as f:
    # with open('word2vec_chainer.model', 'r') as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            word2index[word] = i
            index2word[i] = word
            # w[i] = numpy.array([float(s.translate(None, "'").translate(None, 'b')) for s in ss[1:]], dtype=numpy.float32)
            w[i] = numpy.array([float(re.sub(r'\'|b', '', s)) for s in ss[1:]], dtype=numpy.float32)
            # w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)

    s = numpy.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))  # normalize

    try:
        while True:
            q = six.moves.input('>> ')
            if q not in word2index:
                print('"{0}" is not found'.format(q))
                continue
            v = w[word2index[q]]
            similarity = w.dot(v)
            print('query: {}'.format(q))
            count = 0
            for i in (-similarity).argsort():
                if numpy.isnan(similarity[i]):
                    continue
                if index2word[i] == q:
                    continue
                print('{0}: {1}'.format(index2word[i], similarity[i]))
                count += 1
                if count == n_result:
                    break

    except EOFError:
        pass
