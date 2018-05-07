# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main DrQA reader training script."""

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

from model import DocumentEncoder
from root import tokenizers
from root.retriever.doc_ranker import DocRanker
from root.retriever.doc_db import DocDB
from root.retriever.tfidf_doc_ranker import TfidfDocRanker

import root.retriever.utils as r_utils
import utils, vector, config, data


PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


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


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    # doc_text = PROCESS_DB.get_doc_text(doc_id)
    text = r_utils.normalize(doc_id)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = r_utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = r_utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    else:
        assert False, 'What is this? {}'.format(match)

    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_texts = answer_doc
    for doc_text in doc_texts:
        if has_answer(answer, doc_text, match):
            return 1
    return 0


MULTIQA_PATH = (
    os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data'),
    PosixPath(__file__).absolute().parents[1].as_posix()
)

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(MULTIQA_PATH[0], 'datasets')
MODEL_DIR = os.path.join(MULTIQA_PATH[1], 'docencoder/results/')
EMBED_DIR = os.path.join(expanduser("~"), 'common/glove/')
READER_PATH = os.path.join(MULTIQA_PATH[1], 
    'reader/results/20180323-a705bbb5.mdl') # pretrained
    # 'reader/results/20180402-20e74b70.mdl') # SQuAD finetuned (v0.2)
    # 'reader/results/20180412-e32835b0.mdl') # TREC finetuned (v0.2)
    # 'reader/results/20180412-d8e66822.mdl') # WebQ finetuned (v0.2)
    # 'reader/results/20180412-69e73c93.mdl') # WikiM finetuned (v0.2)

    # 'reader/results/20180402-20e74b70.mdl') # SQuAD finetuned (v0.1)
    # 'reader/results/20180402-b6765c55.mdl') # TREC finetuned (v0.1)
    # 'reader/results/20180402-14e759b0.mdl') # WebQ finetuned (v0.1)
    # 'reader/results/20180402-018b239a.mdl') # WikiM finetuned (v0.1)

    # 'reader/results/20180402-f6c823ed.mdl') # TREC finetuned (dep)
    # 'reader/results/20180402-072b2054.mdl') # WebQ finetuned (dep)
    # 'reader/results/20180402-37b6e412.mdl') # WikiM finetuned (dep)
RET_PATH = os.path.join(MULTIQA_PATH[0], 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
DOC_DB_PATH = os.path.join(MULTIQA_PATH[0], 'wikipedia/docs.db')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/SQuAD-v1.1-valid.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/CuratedTrec-valid.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/WebQuestions-valid.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/WikiMovies-valid.txt')

QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/SQuAD-v1.1-dev.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/CuratedTrec-test.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/WebQuestions-test.txt')
# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/WikiMovies-test.txt')
DOCENCODER_PATH = os.path.join(MULTIQA_PATH[1], 
    # 'docencoder/results/20180323-8d3fa60d.mdl') # soft + simple (pretrained)
    # 'docencoder/results/20180404-e6869910.mdl') # SQuAD finetuned
    # 'docencoder/results/20180404-c0308e5e.mdl') # TREC finetuned
    # 'docencoder/results/20180323-85eea1e4.mdl') # WebQ finetuned
    # 'docencoder/results/20180323-205d5338.mdl') # WikiM finetuned
    # 'docencoder/results/20180329-849cb50b.mdl') # soft + hard + simple
    'docencoder/results/20180419-f5c79d9a.mdl') # soft + hard (v0.2)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
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
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=20,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--predict-batch-size', type=int, default=100,
                         help='Batch size for question prediction')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--retriever-name', type=str, default=RET_PATH,
                       help='Unique retriever path')
    files.add_argument('--reader-name', type=str, default=READER_PATH,
                       help='Unique reader path')
    files.add_argument('--doc-db', type=str, default=DOC_DB_PATH,
                       help='Unique doc db path')
    files.add_argument('--query-data', type=str, default=QUERY_PATH,
                       help='Unique query path')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-processed-simple.txt',
                       # default='distant/SQuAD-v1.1-train.dstrain',
                       # default='distant/CuratedTrec-train.dstrain',
                       # default='distant/WebQuestions-train.dstrain',
                       # default='distant/WikiMovies-train.dstrain',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-simple.txt',
                       # default='distant/SQuAD-v1.1-train.dsdev',
                       # default='distant/CuratedTrec-train.dsdev',
                       # default='distant/WebQuestions-train.dsdev',
                       # default='distant/WikiMovies-train.dsdev',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--dev-txt', type=str, default='SQuAD-v1.1-dev.txt',
    # files.add_argument('--dev-txt', type=str, default='CuratedTrec-test.txt',
                       help=('QA pairs of dev file'))
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')
    files.add_argument('--candidate-file', type=str, default=None,
                        help=("List of candidates to restrict predictions to, "
                              "one candidate per line"))
    files.add_argument('--no-embed', type='bool', default=False,
                       help='Do not load pretrained for faster debug')
    files.add_argument('--tokenizer', type=str, default='regexp',
                       help=('Tokenizer'))
    files.add_argument('--match', type=str, default='string',
                       help=('Matching function'))
    files.add_argument('--n-docs', type=int, default=5,
                       help=('Number of documents for filtering'))
    files.add_argument('--n-pars', type=int, default=10,
                       help=('Number of paragraphs'))

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=DOCENCODER_PATH,
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--neg-size', type=int, default=5,
                            help='Number of negative samples')
    preprocess.add_argument('--allowed-size', type=int, default=100,
                            help='Length allowed for the same batch')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='P@5',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=500,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError('No such file: %s' % args.dev_json)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
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

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    model = DocumentEncoder(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file and not args.no_embed:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.3f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def encode_docs_qs(args, data_loader, model, global_stats, mode):
    """Encode given documents.
    """
    eval_time = utils.Timer()

    # Encode documents
    examples = 0
    documents = []
    questions = []
    for ex in data_loader:
        docs, qs = model.encode(ex)
        batch_size = ex[0].size(0)
        examples += batch_size

        # Save encoded documents and questions
        documents.append(docs)
        questions.append(qs)

    documents = torch.cat(documents, 0)
    questions = torch.cat(questions, 0)
    logger.info('%s valid encoding: Epoch = %d | encoded = %d | ' %
                (mode, global_stats['epoch'], documents.size(0)) +
                'examples = %d | ' % (examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return documents, questions


def rank_docs(args, documents, questions, global_stats, mode, top_k=5):
    """Rank documents given dev questions."""
    eval_time = utils.Timer()
    p1 = utils.AverageMeter()
    p5 = utils.AverageMeter()

    # Iterate documents and check if in top k
    for q_idx, q in enumerate(questions):
        doc_scores = torch.matmul(documents, q.unsqueeze(1)).squeeze()
        topk_docs = torch.topk(doc_scores, top_k)[1]
        
        p1.update(q_idx == topk_docs.data.tolist()[0])
        p5.update(q_idx in topk_docs.data.tolist())

    logger.info('%s valid result: Epoch = %d | P@1 = %.3f | P@5 = %.3f | ' %
                (mode, global_stats['epoch'], p1.avg, p5.avg) +
                'examples = %d | ' % len(questions) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'P@1': p1.avg, 'P@5': p5.avg}


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # If we are doing offician evals then we need to:
    # 1) Load the original text to retrieve spans from offsets.
    # 2) Load the (multiple) text answers for each question.
    if args.official_eval:
        dev_texts = utils.load_text(args.dev_json)
        dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
        dev_answers = utils.load_answers(args.dev_json)

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocumentEncoder.load_checkpoint(checkpoint_file, 
                                                             args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocumentEncoder.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, 
                                       args.neg_size, args.allowed_size)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.ae_train_batchify,
        pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model,
                                     neg_size=1, allowed_size=1000)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.ae_dev_batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Filtering by questions
        # pre_selected_docs = filter_docs(args, dev_loader)

        # Encode documents for dev
        docs, qs = encode_docs_qs(args, dev_loader, model, stats, mode='dev')

        # Rank encoded documents
        result = rank_docs(args, docs, qs, stats, mode='dev')

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.3f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]

    # Encoder final evaluation
    docs, qs = encode_docs_qs(args, dev_loader, model, stats, mode='dev')
    result = rank_docs(args, docs, qs, stats, mode='dev')

    # --------------------------------------------------------------------------
    # EVAL WITH DOCUMENT ENCODER
    start = time.time()
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.query_data):
        qa_pair = json.loads(line)
        question = qa_pair['question']
        answer = qa_pair['answer']
        questions.append(question)
        answers.append(answer)

    # Load candidates
    if args.candidate_file:
        logger.info('Loading candidates from %s' % args.candidate_file)
        candidates = set()
        with open(args.candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
        logger.info('Loaded %d candidates.' % len(candidates))
    else:
        candidates = None

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    tok_class = tokenizers.get_class(args.tokenizer)
    # ranker = TfidfDocRanker(tfidf_path=args.retriever_name)
    ranker = DocRanker(model, args.reader_name, db_path=args.doc_db,
                       tfidf_path=args.retriever_name,
                       fixed_candidates=candidates)

    logger.info('Ranking...')
    '''
    closest_pars = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=None
    )
    '''
    # Batcify questions and feed for ranking
    # qas = list(zip(questions, answers))
    # random.shuffle(qas)
    # questions, answers = zip(*qas)
    batches = [questions[i: i + args.predict_batch_size]
               for i in range(0, len(questions), args.predict_batch_size)]
               # for i in range(0, args.predict_batch_size, args.predict_batch_size)]
    batches_targets = [answers[i: i + args.predict_batch_size]
                       for i in range(0, len(answers), args.predict_batch_size)]
    closest_pars = []
    with open('predictions_{}.json'.format(args.model_name), 'w') as outf:
        for i, (batch, target) in enumerate(zip(batches, batches_targets)):
            logger.info(
                '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
            )
            closest_par, predictions = ranker.rank_docs(batch, target,
                                                        n_docs=args.n_docs,
                                                        n_pars=args.n_pars)
            closest_pars += closest_par
            for p in predictions:
                outf.write(json.dumps(p) + '\n')

    # answers_docs = zip(answers[:args.predict_batch_size], 
    #                    closest_pars[:args.predict_batch_size])
    answers_docs = zip(answers, closest_pars)

    # define processes
    tok_opts = {}
    db_class = DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.data_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

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
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
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

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
