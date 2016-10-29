#'!/usr/bin/env python
#-*- coding:utf-8 -*-
#!/usr/bin/python3

from util.functions import trace
import numpy as np

from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import chainer.links as L

from EncoderDecoderModel import EncoderDecoderModel
import subprocess

from word2vec.word2vec_load import SkipGram,SoftmaxCrossEntropyLoss

unit = 300
vocab = 5000
loss_func = SoftmaxCrossEntropyLoss(unit, vocab)
w2v_model = SkipGram(vocab, unit, loss_func)
serializers.load_hdf5("./word2vec/word2vec_chainer.model", w2v_model)


class EncoderDecoderModelForward(EncoderDecoderModel):

    def forward(self, src_batch, trg_batch, src_vocab, trg_vocab, encdec, is_training, generation_limit):
        batch_size = len(src_batch)
        src_len = len(src_batch[0])
        trg_len = len(trg_batch[0]) if trg_batch else 0
        src_stoi = src_vocab.stoi
        trg_stoi = trg_vocab.stoi
        trg_itos = trg_vocab.itos
        encdec.reset(batch_size)

        x = self.common_function.my_array([src_stoi('</s>') for _ in range(batch_size)], np.int32)
        encdec.encode(x)
        for l in reversed(range(src_len)):
            x = self.common_function.my_array([src_stoi(src_batch[k][l]) for k in range(batch_size)], np.int32)
            encdec.encode(x)

        t = self.common_function.my_array([trg_stoi('<s>') for _ in range(batch_size)], np.int32)
        hyp_batch = [[] for _ in range(batch_size)]

        if is_training:
            loss = self.common_function.my_zeros((), np.float32)
            for l in range(trg_len):
                y = encdec.decode(t)
                t = self.common_function.my_array([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], np.int32)
                loss += functions.softmax_cross_entropy(y, t)
                output = cuda.to_cpu(y.data.argmax(1))
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))
            return hyp_batch, loss

        else:
            while len(hyp_batch[0]) < generation_limit:
                y = encdec.decode(t)
                output = cuda.to_cpu(y.data.argmax(1))
                t = self.common_function.my_array(output, np.int32)
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))
                if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
                    break

        return hyp_batch

parameter_dict = {}
train_path = "Data/"
parameter_dict["source"] = train_path + "player_1_wakati"
parameter_dict["target"] = train_path + "player_2_wakati"
parameter_dict["test_source"] = train_path + "player_1_wakati"
parameter_dict["test_target"] = train_path + "player_2_test"
#--------Hands on  2----------------------------------------------------------------

"""
下記の値が大きいほど扱える語彙の数が増えて表現力が上がるが計算量が爆発的に増えるので大きくしない方が良いです。
"""
parameter_dict["vocab"] = 5000

"""
この数が多くなればなるほどモデルが複雑になります。この数を多くすると必然的に学習回数を多くしないと学習は
収束しません。
語彙数よりユニット数の数が多いと潜在空間への写像が出来ていないことになり結果的に意味がない処理になります。
"""
parameter_dict["embed"] = 300

"""
この数も多くなればなるほどモデルが複雑になります。この数を多くすると必然的に学習回数を多くしないと学習は
収束しません。
"""
parameter_dict["hidden"] = 500

"""
学習回数。基本的に大きい方が良いが大きすぎると収束しないです。
"""
parameter_dict["epoch"] = 20

"""
ミニバッチ学習で扱うサイズです。この点は経験的に調整する場合が多いが、基本的に大きくすると学習精度が向上する
代わりに学習スピードが落ち、小さくすると学習精度が低下する代わりに学習スピードが早くなります。
"""
parameter_dict["minibatch"] = 64

"""
予測の際に必要な単語数の設定。長いほど多くの単語の翻訳が確認できるが、一般的にニューラル翻訳は長い翻訳には
向いていないので小さい数値がオススメです。
"""
parameter_dict["generation_limit"] = 256

parameter_dict["word2vec"] = w2v_model

parameter_dict["word2vecFlag"] = True


parameter_dict["encdec"] = ""

#--------Hands on  2----------------------------------------------------------------#

trace('initializing ...')

encoderDecoderModel = EncoderDecoderModelForward(parameter_dict)
# encoderDecoderModel.train()
encoderDecoderModel.test()
