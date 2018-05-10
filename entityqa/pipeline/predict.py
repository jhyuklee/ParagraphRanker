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
from root.ranker.model import DocumentEncoder
from root.pipeline.pipeline import QAPipeline
from root.retriever.doc_db import DocDB

import root.retriever.utils as r_utils


MULTIQA_PATH = (
    os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data'),
    PosixPath(__file__).absolute().parents[1].as_posix()
)

logger = logging.getLogger()

# Defaults
DATA_DIR = os.path.join(MULTIQA_PATH[0], 'datasets')
PIPELINE_DIR = os.path.join(MULTIQA_PATH[1], 'pipeline/results/')
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
RANKER_PATH = os.path.join(MULTIQA_PATH[1], 
    # 'ranker/results/20180323-8d3fa60d.mdl') # soft + simple (pretrained)
    # 'ranker/results/20180404-e6869910.mdl') # SQuAD finetuned
    # 'ranker/results/20180404-c0308e5e.mdl') # TREC finetuned
    # 'ranker/results/20180323-85eea1e4.mdl') # WebQ finetuned
    # 'ranker/results/20180323-205d5338.mdl') # WikiM finetuned
    # 'ranker/results/20180329-849cb50b.mdl') # soft + hard + simple
    # 'ranker/results/20180419-f5c79d9a.mdl') # soft + hard (v0.2)
    'ranker/results/20180509-66a734e9.mdl') # test


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
    runtime.add_argument('--predict-batch-size', type=int, default=100,
                         help='Batch size for question prediction')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=PIPELINE_DIR,
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
    files.add_argument('--n-docs', type=int, default=20,
                       help=('Number of documents for filtering'))
    files.add_argument('--n-pars', type=int, default=200,
                       help=('Number of paragraphs'))

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=RANKER_PATH,
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

    return args


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


def main(args):
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
    logger.info('Initializing pipeline...')
    tok_class = tokenizers.get_class(args.tokenizer)
    # ranker = TfidfDocRanker(tfidf_path=args.retriever_name)
    pipeline = QAPipeline(args.pretrained, args.reader_name, 
                          db_path=args.doc_db,
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
    with open(os.path.join(MULTIQA_PATH[1], args.model_dir,
              'predictions_{}.json'.format(args.model_name)), 'w') as outf:
        for i, (batch, target) in enumerate(zip(batches, batches_targets)):
            logger.info(
                '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
            )
            closest_par, predictions = pipeline.rank_docs(batch, target,
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

    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # Run!
    main(args)
