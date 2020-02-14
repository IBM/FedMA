import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        
        #output = self.drop(output)
        
        #logger.info("Output size: {}, what if we do the same thing as tf: {}".format(output.size(), output[-1,:,:].size())) 
        #logger.info("Shape of decoder : {}".format(self.decoder))       

        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder(output[-1,:,:])
        #logger.info("Done decoded ...")
        #logger.info("######## Shape of decoded: {}".format(decoded.size()))
        return decoded.t(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
            #return (weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)



class RNNModelContainer(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, rnn_type, ntoken, ninp_l1, nhid_l1, ninp_l2, nhid_l2, dropout=0.5, tie_weights=False):
        super(RNNModelContainer, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp_l1)
        
        self.rnn1 = nn.LSTM(ninp_l1, nhid_l1, 1)

        assert ninp_l2 == nhid_l1 # these two shapes are designed to be equal
        self.rnn2 = nn.LSTM(ninp_l2, nhid_l2, 1)

        self.decoder = nn.Linear(nhid_l2, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid_l1 = nhid_l1
        self.nhid_l2 = nhid_l2

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        output1, hidden1 = self.rnn1(emb, hidden[0])
        
        output2, hidden2 = self.rnn2(output1, hidden[1])
        #output = self.drop(output)
        
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder(output2[-1,:,:])

        return decoded.t(), (hidden1, hidden2)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return ((weight.new_zeros(1, bsz, self.nhid_l1),
                    weight.new_zeros(1, bsz, self.nhid_l1)),
                    (weight.new_zeros(1, bsz, self.nhid_l2),
                    weight.new_zeros(1, bsz, self.nhid_l2)))