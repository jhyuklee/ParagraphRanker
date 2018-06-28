import argparse
import torch
import numpy as np
import regex as re
import json
import os
import sys
import subprocess
import logging
import time
import random

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from pathlib import PosixPath
from os.path import expanduser

from root import tokenizers
from root.ranker import data, utils
from root.pipeline.pipeline import QAPipeline
from root.retriever.doc_db import DocDB
from root.retriever.eval import init, regex_match, has_answer, get_score 
import root.retriever.utils as r_utils

logger = logging.getLogger()

'''
Usage example:
    ~/EntityQA $ python entityqa/pipeline/predict.py \
                        --query-type SQuAD \
                        --ranker-type default \
                        --reader-type default
'''

# (root/data, root/entityqa) paths
ROOT_PATH = PosixPath(__file__).absolute().parents[2].as_posix()
PATHS = {
    'DATA': os.path.join(ROOT_PATH, 'data/datasets'),
    'EMBED': os.path.join(expanduser("~"), 'datasets'),
    'PIPELINE': os.path.join(ROOT_PATH, 'entityqa/pipeline/results'),
    'WIKI': os.path.join(ROOT_PATH, 'data/wikipedia'),
    'RANKER': os.path.join(ROOT_PATH, 'entityqa/ranker/results'),
    'READER': os.path.join(ROOT_PATH, 'entityqa/reader/results'),
}

# Retriever, DB paths
RET_PATH = os.path.join(PATHS['WIKI'],
    'docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
DOC_DB_PATH = os.path.join(PATHS['WIKI'], 'docs.db')

# Reader paths (default + fine-tined)
READER_PATH = {
    'default': os.path.join(PATHS['READER'], '20180613-a4fbf221.mdl'),
    'SQuAD': os.path.join(PATHS['READER'], '20180402-20e74b70.mdl'),
    'TREC': os.path.join(PATHS['READER'], '20180613-a4fbf221.mdl'),
    'WebQ': os.path.join(PATHS['READER'], '20180402-14e759b0.mdl'),
    'WikiM': os.path.join(PATHS['READER'], '20180402-018b239a.mdl'),
}

# Ranker paths
RANKER_PATH = {
    'default': os.path.join(PATHS['RANKER'], '20180613-a6982591.mdl'),
    'TREC': os.path.join(PATHS['RANKER'], '20180618-3f538bf0.ranker'),
    'soft': os.path.join(PATHS['RANKER'], '20180613-a6982591.mdl'),
    'hard': os.path.join(PATHS['RANKER'], '20180419-f5c79d9a.mdl'),
}

# Query path
DATASETS = {
    'SQuAD': 'SQuAD-v1.1',
    'TREC': 'CuratedTrec',
    'WebQ': 'WebQuestions',
    'WikiM': 'WikiMovies'
}
QUERY_PATH = (lambda dataset, split: \
    os.path.join(PATHS['DATA'], '{}-{}.txt'.format(DATASETS[dataset], split)))

# Candidate paths
CANDIDATES = {
    'WebQ': os.path.join(PATHS['DATA'], 'freebase-entities.txt'),
    'WikiM': os.path.join(PATHS['DATA'], 'WikiMovies-entities.txt'),
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--train', type='bool', default=False,
                        help='if train modules')
    runtime.add_argument('--train-epoch', type=int, default=30,
                         help='number of training epoch')
    runtime.add_argument('--predict-batch-size', type=int, default=100,
                         help='Batch size for question prediction')
    runtime.add_argument('--module-batch-size', type=int, default=64,
                         help='Batch size for reader/ranker in pipeline')
    runtime.add_argument('--module-max-loaders', type=int, default=5,
                         help='Max num loaders for reader/ranker dataset')
    runtime.add_argument('--n-docs', type=int, default=20,
                         help=('Number of documents for filtering'))
    runtime.add_argument('--n-pars', type=int, default=200,
                         help=('Number of paragraphs'))
    runtime.add_argument('--tokenizer', type=str, default='regexp',
                         help=('Tokenizer'))
    runtime.add_argument('--match', type=str, default='string',
                         help=('Matching function (set to regex for Trec)'))
    runtime.add_argument('--ranker-type', type=str, default='default',
                         help=('Type of ranker'))
    runtime.add_argument('--reader-type', type=str, default='TREC',
                         help=('Type of reader'))
    runtime.add_argument('--query-type', type=str, default='TREC',
                         help=('Type of query'))
    runtime.add_argument('--query-split', type=str, default='test',
                         help=('Split type of query (valid/test)'))

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=PATHS['PIPELINE'],
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--retriever-path', type=str, default=RET_PATH,
                       help='Unique retriever path')
    files.add_argument('--ranker-path', type=str, default=RANKER_PATH['default'],
                       help='Unique ranker path')
    files.add_argument('--reader-path', type=str, default=READER_PATH['default'],
                       help='Unique reader path')
    files.add_argument('--db-path', type=str, default=DOC_DB_PATH,
                       help='Unique doc db path')
    files.add_argument('--query-data', type=str, default=QUERY_PATH('SQuAD', 'dev'),
                       help='Unique query path')
    files.add_argument('--embed-dir', type=str, default=PATHS['EMBED'],
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       # default='glove/glove.840B.300d.txt',
                       default='fasttext/crawl-300d-2M.vec',
                       help='Space-separated pretrained embeddings file')
    files.add_argument('--fasttext', type='bool', default=True,
                        help='if using fasttext')
    files.add_argument('--candidate-file', type=str, default=None,
                        help=("List of candidates to restrict predictions to, "
                              "one candidate per line"))


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set logging
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.ranker_file = os.path.join(args.model_dir, args.model_name + '.ranker')
    args.pred_file = os.path.join(args.model_dir, 
                                  'pred_{}.json'.format(args.model_name))
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info('-' * 100)

    # Embeddings options
    if args.embedding_file:
        if not args.fasttext:
            with open(args.embedding_file) as f:
                dim = len(f.readline().strip().split(' ')) - 1
            args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Set ranker/reader/query paths
    if args.ranker_type:
        args.ranker_path = RANKER_PATH[args.ranker_type]
    if args.reader_type:
        args.reader_path = READER_PATH[args.reader_type]
    if args.query_type:
        if args.query_type == 'SQuAD' and args.query_split == 'test':
            args.query_split = 'dev'
        args.query_data = QUERY_PATH(args.query_type, args.query_split)
    if args.query_type == 'TREC':
        args.match = 'regex'
        logger.info('{} match for {}'.format(args.match, args.query_type))
    args.candidate_file = CANDIDATES.get(args.query_type, None)
    
    return args


def main(args):
    # Read query data
    start = time.time()
    logger.info('Reading query data {}'.format(args.query_data))
    questions = []
    answers = []
    for line in open(args.query_data):
        qa_pair = json.loads(line)
        question = qa_pair['question']
        answer = qa_pair['answer']
        questions.append(question)
        answers.append(answer)

    # Load candidates
    candidates = None
    if args.candidate_file:
        logger.info('Loading candidates from %s' % args.candidate_file)
        candidates = set()
        with open(args.candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
        logger.info('Loaded %d candidates.' % len(candidates))

    # get the closest docs for each question.
    logger.info('Initializing pipeline...')
    pipeline = QAPipeline(retriever_path=args.retriever_path,
                          db_path=args.db_path,
                          ranker_path=args.ranker_path,
                          reader_path=args.reader_path,
                          module_batch_size=args.module_batch_size,
                          module_max_loaders=args.module_max_loaders,
                          module_cuda=args.cuda,
                          fixed_candidates=candidates)

    # Batcify questions and feed for prediction
    batches = [questions[i: i + args.predict_batch_size]
        for i in range(0, len(questions), args.predict_batch_size)]
    batches_targets = [answers[i: i + args.predict_batch_size]
        for i in range(0, len(answers), args.predict_batch_size)]

    # Predict and record results
    logger.info('Predicting...' if not args.train else 'Training...')
    if args.train:
        best_loss = 9999999
        train_loss = utils.AverageMeter()
        for e_idx in range(args.train_epoch):
            logger.info('Epoch {}'.format(e_idx+1))
            for i, (batch, target) in enumerate(zip(batches, batches_targets)):
                logger.info(
                    '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
                )
                loss = pipeline.update(batch, target, 
                                       n_docs=args.n_docs, n_pars=args.n_pars)
                train_loss.update(loss, 1)
                
            if train_loss.avg < best_loss:
                logger.info('Best loss = %.3f' % train_loss.avg)
                pipeline.ranker.save(args.ranker_file)
                best_loss = train_loss.avg

        logger.info('Training done')
        exit()
    else:
        closest_pars = []
        with open(args.pred_file, 'w') as pred_f:
            for i, (batch, target) in enumerate(zip(batches, batches_targets)):
                logger.info(
                    '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
                )
                with torch.no_grad():
                    closest_par, predictions = pipeline.predict(batch,
                                                                n_docs=args.n_docs,
                                                                n_pars=args.n_pars)
                    closest_pars += closest_par

                for p in predictions:
                    pred_f.write(json.dumps(p) + '\n')

            answers_pars = zip(answers, closest_pars)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = DocDB
    db_opts = {'db_path': args.db_path}
    processes = ProcessPool(
        processes=args.data_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match, use_text=True)
    scores = processes.map(get_score_partial, answers_pars)

    filename = os.path.basename(args.query_data)
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


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'EntityQA Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # PRINT CONFIG
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # Run!
    main(args)
