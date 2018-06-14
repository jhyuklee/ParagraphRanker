#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Follows official evaluation script for v1.1 of the SQuAD dataset."""

import argparse
import json
import operator
import numpy as np
from collections import Counter
from root.retriever.utils import normalize
from root.reader.utils import (
    exact_match_score,
    f1_score,
    regex_match_score,
    metric_max_over_ground_truths
)


def evaluate(dataset_file, prediction_file, regex=False, top_k=None, alpha=1.0, beta=1.0, gamma=1.0):
    print('-' * 50)
    print('Alpha {}, Beta {}, Gamma {}'.format(alpha, beta, gamma), end=' ')

    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    predictions = []
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)

            # Set top_k as all predictions
            if top_k is None:
                top_k = len(data)
                print('Evaluating top {} answers'.format(top_k))

            # Accumulate all predictions
            top_pred = []
            top_ascore = {}
            top_pscore = {}
            top_dscore = {}
            for single_pred in data[:top_k]:
                prediction = normalize(single_pred['span'])
                top_pred.append(prediction)

                # Score based rank
                if prediction not in top_ascore:
                    top_ascore[prediction] = 0
                    top_pscore[prediction] = 0
                    top_dscore[prediction] = 0
                
                # alpha = 1.0
                # beta = 1.0
                # gamma = 1.0
                top_ascore[prediction] += (single_pred['span_score'] ** alpha
                                           if single_pred['span_score'] > 0
                                           else 0)
                top_pscore[prediction] += (single_pred['sidx_score'] ** beta
                                           if single_pred['sidx_score'] > 0
                                           else 0)
                top_dscore[prediction] += (single_pred['doc_score'] ** gamma
                                           if single_pred['doc_score'] > 0
                                           else 0)
                # top_ascore[prediction] += single_pred['span_score']
                # top_pscore[prediction] += single_pred['sidx_score']
                # top_dscore[prediction] += single_pred['doc_score']

            # Perform normalization (softmax or normalize)
            sorted_atop = sorted(top_ascore.items(), key=operator.itemgetter(1),
                                 reverse=True)[:]
            sorted_ptop = sorted(top_pscore.items(), key=operator.itemgetter(1),
                                 reverse=True)[:]
            sorted_dtop = sorted(top_dscore.items(), key=operator.itemgetter(1),
                                 reverse=True)[:]

            amax = sorted_atop[0][1]
            # atop_sum = sum([np.exp(p[1]-amax) for p in sorted_atop])
            # sorted_atop = [(p[0], np.exp(p[1]-amax)/atop_sum) for p in sorted_atop]
            # atop_sum = sum([p[1] for p in sorted_atop])
            # sorted_atop = [(p[0], p[1]) for p in sorted_atop]

            pmax = sorted_ptop[0][1]
            # ptop_sum = sum([np.exp(p[1]-pmax) for p in sorted_ptop])
            # sorted_ptop = [(p[0], np.exp(p[1]-pmax)/ptop_sum) for p in sorted_ptop]
            # ptop_sum = sum([p[1] for p in sorted_ptop])
            # sorted_ptop = [(p[0], p[1]) for p in sorted_ptop]

            dmax = sorted_dtop[0][1]
            # dtop_sum = sum([np.exp(p[1]-dmax) for p in sorted_dtop])
            # sorted_dtop = [(p[0], np.exp(p[1]-dmax)/dtop_sum) for p in sorted_dtop]
            # dtop_sum = sum([p[1] for p in sorted_dtop])
            # sorted_dtop = [(p[0], p[1]) for p in sorted_dtop]

            # Merge scores
            merged_top = dict(sorted_atop)
            for answer, score in dict(sorted_ptop).items():
                # score = 1
                if answer in merged_top:
                    merged_top[answer] *= (score)
                else:
                    merged_top[answer] = score

            for answer, score in dict(sorted_dtop).items():
                # score = 1
                if answer in merged_top:
                    merged_top[answer] *= (score)
                else:
                    merged_top[answer] = score

            sorted_top = sorted(merged_top.items(), key=operator.itemgetter(1),
                                reverse=True)

            top_pred = [p[0] for p in sorted_top[:1]]
            predictions.append(top_pred)

            # Majority voting
            # pred = Counter(top_pred).most_common(1)
            # predictions.append([p[0] for p in pred])

    exact_match = 0
    f1 = 0
    for i in range(len(predictions)):
        match_fn = regex_match_score if regex else exact_match_score
        single_em = 0
        # single_f1 = 0
        assert len(predictions[i]) == 1
        for single_pred in predictions[i]:
            single_em = metric_max_over_ground_truths(
                match_fn, single_pred, answers[i]
            )
            # single_f1 = metric_max_over_ground_truths(
            #     f1_score, single_pred, answers[i]
            # )
        assert single_em == 1 or single_em == 0
        # assert single_f1 >= 0 and single_f1 <= 1
        exact_match += single_em
        # f1 += single_f1

        '''
        if single_em > 0:
            print(answers[i])
            print(predictions[i])
            print('match\n')
        else:
            print()
        '''
    total = len(predictions)
    exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total
    print({'exact_match': exact_match})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/datasets/SQuAD-v1.1-dev.txt')
    # parser.add_argument('--dataset', type=str, default='../data/datasets/CuratedTrec-test.txt')
    # parser.add_argument('--dataset', type=str, default='../data/datasets/WebQuestions-test.txt')
    # parser.add_argument('--dataset', type=str, default='../data/datasets/WikiMovies-test.txt')
    parser.add_argument('--predictions', type=str, default='predictions.json')
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()

    alphas = list(range(1, 2))
    betas = list(range(1, 2))
    gammas = list(range(1, 2))

    print('Dataset: %s' % args.dataset)
    print('Predictions: %s' % args.predictions)

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                evaluate(args.dataset, args.predictions, args.regex, args.top_k,
                         alpha, beta, gamma)
