#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from pathlib import PosixPath
from os.path import expanduser

from root.retriever.tfidf_doc_ranker import TfidfDocRanker 
from root.retriever.doc_db import DocDB
from root import tokenizers
import root.retriever.utils as r_utils


MULTIQA_PATH = (
    os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data'),
    PosixPath(__file__).absolute().parents[1].as_posix()
)

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS = {
    'TOK': None,
    'DB': None,
    'CANDS': None,
}


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS
    PROCESS['TOK'] = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS['TOK'], PROCESS['TOK'].shutdown, exitpriority=100)
    PROCESS['DB'] = db_class(**db_opts)
    Finalize(PROCESS['DB'], PROCESS['DB'].close, exitpriority=100)
    PROCESS['CANDS'] = candidates


def fetch_text(doc_id):
    global PROCESS
    return PROCESS['DB'].get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS
    return PROCESS['TOK'].tokenize(text)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_data, match, use_text=False):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS

    if use_text:
        text = r_utils.normalize(doc_data)
    else:
        text = fetch_text(doc_data)
        text = r_utils.normalize(text)

    if match == 'string':
        # Answer is a list of possible strings
        text = tokenize_text(text).words(uncased=True)
        for single_answer in answer:
            single_answer = r_utils.normalize(single_answer)
            single_answer = tokenize_text(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = r_utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match, use_text=False):
    """Search through all the top docs to see if they have the answer."""
    if use_text:
        answer, doc_data = answer_doc
    else:
        answer, (doc_data, doc_scores) = answer_doc

    for doc_datum in doc_data:
        if has_answer(answer, doc_datum, match, use_text):
            return 1
    return 0


def get_answer_label(answer_doc, match, use_text=False):
    """Search through all the top docs to see if they have the answer."""
    if use_text:
        answer, doc_data = answer_doc
    else:
        answer, (doc_data, doc_scores) = answer_doc

    answer_labels = []
    for doc_datum in doc_data:
        if has_answer(answer, doc_datum, match, use_text):
            answer_labels += [1.0]
        else:
            answer_labels += [0.0]

    return answer_labels


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    MODEL_PATH = os.path.join(MULTIQA_PATH[0], 
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
    DOC_DB_PATH = os.path.join(MULTIQA_PATH[0], 'wikipedia/docs.db')
    # DATASET_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/SQuAD-v1.1-dev.txt')
    DATASET_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/CuratedTrec-test.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET_PATH)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--doc-db', type=str, default=DOC_DB_PATH,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    # TREC => regex
    # SQuAD => string
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = TfidfDocRanker(tfidf_path=args.model)

    logger.info('Ranking...')
    print('processing query', questions[0])
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=args.num_workers
    )
    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)
