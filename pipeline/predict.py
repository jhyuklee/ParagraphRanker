

# Defaults
DATA_DIR = os.path.join(MULTIQA_PATH[0], 'datasets')
MODEL_DIR = os.path.join(MULTIQA_PATH[1], 'docencoder/results/')
EMBED_DIR = os.path.join(expanduser("~"), 'common/glove/')
READER_PATH = os.path.join(MULTIQA_PATH[1], 
    # 'reader/results/20180323-a705bbb5.mdl') # pretrained
    # 'reader/results/20180402-20e74b70.mdl') # SQuAD finetuned (v0.2)
    # 'reader/results/20180412-e32835b0.mdl') # TREC finetuned (v0.2)
    # 'reader/results/20180412-d8e66822.mdl') # WebQ finetuned (v0.2)
    # 'reader/results/20180412-69e73c93.mdl') # WikiM finetuned (v0.2)

    # 'reader/results/20180402-20e74b70.mdl') # SQuAD finetuned (v0.1)
    'reader/results/20180402-b6765c55.mdl') # TREC finetuned (v0.1)
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

# QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/SQuAD-v1.1-dev.txt')
QUERY_PATH = os.path.join(MULTIQA_PATH[0], 'datasets/CuratedTrec-test.txt')
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
    with open('results/predictions_{}.json'.format(args.model_name), 'w') as outf:
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
