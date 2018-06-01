#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import math
import six

import chainer
import torch
import torch.nn.functional as F

from chainer import reporter

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from e2e_asr_attctc_th import AttLoc


def encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('tanh'))


class Reporter(chainer.Chain):
    def report(self, mse_loss, bce_loss, loss):
        reporter.report({'mse_loss': mse_loss}, self)
        reporter.report({'bce_loss': bce_loss}, self)
        reporter.report({'loss': loss}, self)


def make_mask(lengths, dim=None):
    """FUNCTION TO MAKE BINARY MASK

    Args:
        length (list): list of lengths
        dim (int): # dimension

    Return:
        (torch.ByteTensor) binary mask tensor (B, Lmax, dim)
    """
    batch = len(lengths)
    maxlen = max(lengths)
    if dim is None:
        mask = torch.zeros(batch, maxlen)
    else:
        mask = torch.zeros(batch, maxlen, dim)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    if torch.cuda.is_available():
        return mask.byte().cuda()
    else:
        return mask.byte()


class Reporter(chainer.Chain):
    def report(self, mse_loss, bce_loss, loss):
        chainer.reporter.report({'mse_loss': mse_loss}, self)
        chainer.reporter.report({'bce_loss': bce_loss}, self)
        chainer.reporter.report({'loss': loss}, self)


class Tacotron2Loss(torch.nn.Module):
    """TACOTRON2 LOSS FUNCTION

    :param tacotron2 torch.nn.Module: Tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    """

    def __init__(self, use_masking=True, bce_pos_weight=20.0):
        super(Tacotron2Loss, self).__init__()
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight
        self.reporter = Reporter()

    def forward(self, outputs, targets):
        """TACOTRON2 LOSS CALCULATION

        :return: tacotron2 loss
        :rtype: torch.Tensor
        """
        # parse inputs
        after_outs, before_outs, logits = outputs
        if len(targets) == 3:
            ys, labels, olens = targets
        else:
            ys, labels = targets
            olens = None

        if self.use_masking and olens is not None:
            if self.bce_pos_weight != 1.0:
                weights = ys.new(*labels.size()).fill_(1)
                weights.masked_fill_(labels.eq(1), self.bce_pos_weight)
            else:
                weights = None
            # masking padded values
            mask = make_mask(olens, ys.size(2))
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            # calculate loss
            mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)
            loss = mse_loss + bce_loss
        else:
            # calculate loss
            mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = mse_loss + bce_loss

        self.reporter.report(mse_loss.item(), bce_loss.item(), loss.item())

        return loss


class Tacotron2(torch.nn.Module):
    """TACOTRON2 BASED SEQ2SEQ MODEL CONVERTS CHARS TO FEATURES

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param int adim: the number of dimension of mlp in attention
    :param int aconv_chans: the number of attention conv filter channels
    :param int aconv_filts: the number of attention conv filter size
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_filts=5,
                 econv_chans=512,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_filts=5,
                 postnet_chans=512,
                 adim=512,
                 aconv_chans=32,
                 aconv_filts=15,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 threshold=0.5,
                 maxlenratio=5.0,
                 minlenratio=0.0):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_filts = econv_filts
        self.econv_chans = econv_chans
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans
        self.postnet_filts = postnet_filts
        self.adim = adim
        self.aconv_filts = aconv_filts
        self.aconv_chans = aconv_chans
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # define network modules
        self.enc = Encoder(idim=self.idim,
                           embed_dim=self.embed_dim,
                           elayers=self.elayers,
                           eunits=self.eunits,
                           econv_layers=self.econv_layers,
                           econv_chans=self.econv_chans,
                           econv_filts=self.econv_filts,
                           use_batch_norm=self.use_batch_norm,
                           dropout=self.dropout)
        self.dec = Decoder(idim=self.eunits,
                           odim=self.odim,
                           att=AttLoc(
                               self.eunits,
                               self.dunits,
                               self.adim,
                               self.aconv_chans,
                               self.aconv_filts),
                           dlayers=self.dlayers,
                           dunits=self.dunits,
                           prenet_layers=self.prenet_layers,
                           prenet_units=self.prenet_units,
                           postnet_layers=self.postnet_layers,
                           postnet_chans=self.postnet_chans,
                           postnet_filts=self.postnet_filts,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=self.use_batch_norm,
                           use_concate=self.use_concate,
                           dropout=self.dropout,
                           threshold=self.threshold,
                           maxlenratio=self.maxlenratio,
                           minlenratio=self.minlenratio)
        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)

    def forward(self, xs, ilens, ys):
        """TACOTRON2 FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        """
        maxlen_in = xs.size(-1)
        hs = self.enc(xs, ilens)
        after_outs, before_outs, logits, att_ws = self.dec(hs, ilens, ys, maxlen_in)

        return after_outs, before_outs, logits, att_ws

    def inference(self, x):
        """GENERATE THE SEQUENCE OF FEATURES FROM THE SEQUENCE OF CHARACTERS

        :param tensor x: the sequence of characters (T)
        :return: the sequence of features (L, odim)
        :rtype: tensor
        :return: the sequence of stop probabilities (L)
        :rtype: tensor
        :return: the sequence of attetion weight (L, T)
        :rtype: tensor
        """
        h = self.enc.inference(x)
        outs, probs, att_ws = self.dec.inference(h)

        return outs, probs, att_ws


class Encoder(torch.nn.Module):
    """CHARACTER EMBEDDING ENCODER

    This is the encoder which converts the sequence of characters into
    the sequence of hidden states. The newtwork structure is based on
    that of tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param bool use_batch_norm: whether to use batch normalization
    :param float dropout: dropout rate
    """

    def __init__(self, idim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_chans=512,
                 econv_filts=5,
                 use_batch_norm=True,
                 use_residual=False,
                 dropout=0.5):
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_chans = econv_chans if econv_layers != 0 else -1
        self.econv_filts = econv_filts if econv_layers != 0 else -1
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout = dropout
        # define network layer modules
        self.embed = torch.nn.Embedding(self.idim, self.embed_dim)
        if self.econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for l in six.moves.range(self.econv_layers):
                ichans = self.embed_dim if l == 0 else self.econv_chans
                if self.use_batch_norm:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(self.econv_chans),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
        else:
            self.convs = None
        iunits = econv_chans if self.econv_layers != 0 else self.embed_dim
        self.blstm = torch.nn.LSTM(
            iunits, self.eunits // 2, self.elayers,
            batch_first=True,
            bidirectional=True)

    def forward(self, xs, ilens):
        """CHARACTER ENCODER FORWARD CALCULATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)
        :return: batch of sequences of padded encoder states (B, Tmax, eunits)
        :rtype: torch.Tensor
        """
        xs = self.embed(xs).transpose(1, 2)
        for l in six.moves.range(self.econv_layers):
            if self.use_residual:
                xs += self.convs[l](xs)
            else:
                xs = self.convs[l](xs)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens, batch_first=True)
        # does not work with dataparallel
        # see https://github.com/pytorch/pytorch/issues/7092#issuecomment-388194623
        # self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, _ = pad_packed_sequence(xs, batch_first=True)

        return xs

    def inference(self, x):
        """CHARACTER ENCODER INFERENCE

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence encoder states (T, eunits)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0]


class Decoder(torch.nn.Module):
    """DECODER TO PREDICT THE SEQUENCE OF FEATURES

    This the decoder which generate the sequence of features from
    the sequence of the hidden states. The network structure is
    based on that of the tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param instance att: instance of attetion class
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim, att,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_chans=512,
                 postnet_filts=5,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 threshold=0.5,
                 maxlenratio=5.0,
                 minlenratio=0.0):
        super(Decoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units if prenet_layers != 0 else self.odim
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans if postnet_layers != 0 else -1
        self.postnet_filts = postnet_filts if postnet_layers != 0 else -1
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # define lstm network
        self.lstm = torch.nn.ModuleList()
        for l in six.moves.range(self.dlayers):
            iunits = self.idim + self.prenet_units if l == 0 else self.dunits
            self.lstm += [torch.nn.LSTMCell(iunits, self.dunits)]
        # define prenet
        if self.prenet_layers > 0:
            self.prenet = torch.nn.ModuleList()
            for l in six.moves.range(self.prenet_layers):
                ichans = self.odim if l == 0 else self.prenet_units
                self.prenet += [torch.nn.Sequential(
                    torch.nn.Linear(ichans, self.prenet_units, bias=False),
                    torch.nn.ReLU())]
        else:
            self.prenet = None
        # define postnet
        if self.postnet_layers > 0:
            self.postnet = torch.nn.ModuleList()
            for l in six.moves.range(self.postnet_layers - 1):
                ichans = self.odim if l == 0 else self.postnet_chans
                ochans = self.odim if l == self.postnet_layers - 1 else self.postnet_chans
                if use_batch_norm:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
            ichans = self.postnet_chans if self.postnet_layers != 1 else self.odim
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(self.dropout))]
            else:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.Dropout(self.dropout))]
        else:
            self.postnet = None
        # define projection layers
        iunits = self.idim + self.dunits if self.use_concate else self.dunits
        self.feat_out = torch.nn.Linear(iunits, self.odim, bias=False)
        self.prob_out = torch.nn.Linear(iunits, 1)

    def zero_state(self, hs):
        return hs.data.new(hs.size(0), self.dunits).zero_()

    def forward(self, hs, hlens, ys, maxlen_in=None):
        """DECODER FORWARD CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :param int maxlen_in: maximum length of inputs (only for dataparallel)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        """
        # check hlens type
        if isinstance(hlens, torch.Tensor):
            hlens = hlens.cpu().numpy()

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        outs, logits, att_ws = [], [], []
        for y in ys.transpose(0, 1):
            att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs)]
            logits += [self.prob_out(zcs)]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.stack(outs, dim=2)  # (B, odim, Lmax)
        after_outs = before_outs + self._postnet_forward(before_outs)  # (B, odim, Lmax)
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        # for dataparallel (should be same size in each split batch)
        if maxlen_in is not None and att_ws.size(-1) != maxlen_in:
            att_ws_new = att_ws.new_zeros(att_ws.size(0), ys.size(1), maxlen_in)
            att_ws_new[:, :, :att_ws.size(-1)] = att_ws
            att_ws = att_ws_new

        return after_outs, before_outs, logits, att_ws

    def inference(self, h):
        """GENERATE THE SEQUENCE OF FEATURES FROM ENCODER HIDDEN STATES

        :param tensor h: the sequence of encoder states (T, C)
        :return: the sequence of features (L, D)
        :rtype: tensor
        :return: the sequence of stop probabilities (L)
        :rtype: tensor
        :return: the sequence of attetion weight (L, T)
        :rtype: tensor
        """
        # setup
        assert len(h.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * self.maxlenratio)
        minlen = int(h.size(0) * self.minlenratio)

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        while True:
            # updated index
            idx += 1

            # decoder calculation
            att_c, att_w = self.att(hs, ilens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs)]
            probs += [F.sigmoid(self.prob_out(zcs))[0]]
            prev_out = outs[-1]
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

            # check whether to finish generation
            if (probs[-1] >= self.threshold and idx >= minlen) or idx == maxlen:
                outs = torch.stack(outs, dim=2)  # (1, odim, L)
                outs = outs + self._postnet_forward(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (Lx, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        return outs, probs, att_ws

    def _prenet_forward(self, x):
        if self.prenet is not None:
            for l in six.moves.range(self.prenet_layers):
                x = F.dropout(self.prenet[l](x), self.dropout)
        return x

    def _postnet_forward(self, xs):
        if self.postnet is not None:
            for l in six.moves.range(self.postnet_layers):
                xs = self.postnet[l](xs)
        return xs
