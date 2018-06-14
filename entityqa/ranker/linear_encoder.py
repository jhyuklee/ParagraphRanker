#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import root.ranker.layers as layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class LinearEncoder(nn.Module):
    def __init__(self, args, normalize=True):
        super(LinearEncoder, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to Linear: word emb + question emb + manual features
        # doc_input_size = args.embedding_dim + args.num_features
        doc_input_size = args.embedding_dim

        # Linear document encoder
        self.doc_linear = nn.Linear(
            in_features=doc_input_size,
            out_features=args.hidden_size,
        )

        # Linear question encoder
        self.question_linear = nn.Linear(
            in_features=args.embedding_dim,
            out_features=args.hidden_size,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = args.hidden_size
        question_hidden_size = args.hidden_size

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for document scores
        # self.doc_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     question_hidden_size,
        # )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Bag of words + embedding
        x1_emb = torch.sum(x1_emb, dim=1)
        x2_emb = torch.sum(x2_emb, dim=1)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        # drnn_input = [x1_emb]

        # Add manual features
        # if self.args.num_features > 0:
        #     drnn_input.append(x1_f)

        # Encode document with RNN
        # doc_hidden = self.doc_linear(torch.cat(drnn_input, 2))
        doc_hidden = self.doc_linear(x1_emb)

        # Encode question with RNN + merge hiddens
        question_hidden = self.question_linear(x2_emb)

        # Predict document scores
        doc_scores = F.sigmoid(torch.sum(doc_hidden * question_hidden, dim=-1))
        
        return doc_scores, doc_hidden, question_hidden
