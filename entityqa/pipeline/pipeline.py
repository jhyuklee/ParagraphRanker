#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import os
import re
import heapq
import math
import time
import torch
import logging
import regex
import operator
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial

from root import tokenizers
from root.retriever import utils
from root.retriever.doc_db import DocDB
from root.ranker.model import DocumentEncoder
from root.ranker.data import ReaderDataset, SortedBatchSampler
from root.ranker.vector import ae_dev_batchify
from root.reader.model import DocReader
from root.reader.data import ReaderDataset as ReaderDataset2
from root.reader.data import SortedBatchSampler as SortedBatchSampler2
from root.reader.vector import batchify


logger = logging.getLogger(__name__)


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


class QAPipeline(object):
    GROUP_LENGTH = 0
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, retriever_path, db_path, ranker_path, reader_path, strict=True,
                 fixed_candidates=None):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        logger.info('Loading %s' % retriever_path)
        matrix, metadata = utils.load_sparse_csr(retriever_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict
        
        # Manually set
        self.max_loaders = 5
        self.cuda = True
        self.batch_size = 64

        logger.info('Loading pretrained ranker...')
        self.ranker = DocumentEncoder.load(ranker_name)
        self.ranker.cuda()
        self.ranker.init_optimizer()

        logger.info('Initializing tokenizers and document retrievers...')
        # tok_class = tokenizers.CoreNLPTokenizer
        tok_class = tokenizers.SimpleTokenizer
        annotators = tokenizers.get_annotators_for_model(self.ranker)
        tok_opts = {'annotators': annotators}

        db_config = {'options': {'db_path': db_path}}
        db_class = db_config.get('class', DocDB)
        db_opts = db_config.get('options', {})

        self.fixed_candidates = fixed_candidates
        self.num_workers = 5
        self.processes = ProcessPool(
            self.num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, self.fixed_candidates)
        )

        logger.info('Loading pretrained reader...')
        self.reader = DocReader.load(reader_name, normalize=False)
        self.reader.cuda()
        self.reader.init_optimizer()


    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def _get_encoder_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.ranker, 1, 1000)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=ae_dev_batchify,
            pin_memory=self.cuda,
        )

        return loader

    def _get_reader_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset2(data, self.reader)
        sampler = SortedBatchSampler2(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )

        return loader

    def predict(self, queries, answers, candidates=None, n_docs=5, n_pars=10):
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)
        
        '''
        # Filter by question
        flat_splits = []
        filtered_texts = []
        overlap = 0
        noverlap = 0
        for q_idx, (docids, query) in enumerate(zip(all_docids, queries)):
            if q_idx < 70: continue
            doc_texts = self.processes.map(fetch_text, docids)
            q_token = self.parse(query)
            # print(q_token)
            for doc_text, docid in zip(doc_texts, docids):
                splits = self._split_doc(doc_text)
                for kk, split in enumerate(splits):
                    # print(split)
                    if any(len(re.findall(r'\b' + x + r'(\:|\,|\.|\?|\b)', 
                           doc_text)) != 0 for x in q_token):
                        overlap += 1
                    else:
                        noverlap += 1
                didx2sidx.append([len(flat_splits), -1])
                for split in splits:
                    flat_splits.append(split)
                didx2sidx[-1][1] = len(flat_splits)
        '''

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        for text in doc_texts:
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)
        logger.info('Tokenizing {} splits'.format(len(flat_splits)))

        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()

        # logger.info('end tokenize {}'.format(len(s_tokens)))

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        for qidx in range(len(queries)):
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(q_tokens[qidx].words()) > 0 and
                            len(s_tokens[sidx].words()) > 0):
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': q_tokens[qidx].words(),
                            # 'qlemma': q_tokens[qidx].lemmas(),
                            'document': s_tokens[sidx].words(),
                            # 'lemma': s_tokens[sidx].lemmas(),
                            # 'pos': s_tokens[sidx].pos(),
                            # 'ner': s_tokens[sidx].entities(),
                        })
                    else:
                        assert False, 'Zero?'

        logger.info('Ranking %d paragraphs...' % len(examples))

        # Push all examples through the document encoder.
        # We decode argmax start/end indices asychronously on CPU.
        ex_ids = []
        sims = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_encoder_loader(examples, num_loaders):
            doc_encoded, q_encoded = self.ranker.encode(batch)
            sim = torch.sum(torch.mul(doc_encoded, q_encoded), dim=1)
            sims.append(sim)
            ex_ids += batch[-1]

        # Calculate pairwise similarity
        similarity = torch.cat(sims, dim=0)

        # Gather scores per questions
        q_gather = {}
        qs_to_score = {}
        for sim, ex_id in zip(similarity, ex_ids): 
            qidx, rel_didx, sidx = ex_id
            doc_sim = float(all_doc_scores[qidx][rel_didx])
            par_sim = sim.data.tolist()[0]

            if qidx not in q_gather:
                q_gather[qidx] = []
            q_gather[qidx] += [[sidx, par_sim, rel_didx, doc_sim, doc_sim * par_sim]]

            # Search score by q and s
            if qidx not in qs_to_score:
                qs_to_score[qidx] = {}
            qs_to_score[qidx][sidx] = sim.data.tolist()[0]

        # Filter only top ranked paragraphs
        n_paragraphs = n_pars
        q_best = {}
        for qidx, pairs in q_gather.items():
            pairs.sort(key=operator.itemgetter(1), reverse=True)
            q_best[qidx] = pairs[:n_paragraphs]

        # Untokenize paragraphs
        paragraphs = []
        for qidx in range(len(queries)):
            paragraphs += [[s_tokens[k[0]].untokenize() for k in q_best[qidx]]]

        # Ready for reader inputs
        reader_examples = []
        for qidx in range(len(queries)):
            for sidx, _, rel_didx, _, _ in q_best[qidx]:
                if (len(q_tokens[qidx].words()) > 0 and
                        len(s_tokens[sidx].words()) > 0):
                    reader_examples.append({
                        'id': (qidx, rel_didx, sidx),
                        'question': q_tokens[qidx].words(),
                        # 'qlemma': q_tokens[qidx].lemmas(),
                        'document': s_tokens[sidx].words(),
                        # 'lemma': s_tokens[sidx].lemmas(),
                        # 'pos': s_tokens[sidx].pos(),
                        # 'ner': s_tokens[sidx].entities(),
                        # 'answers': 'hihi'
                    })
                else:
                    assert False, 'Zero length words'

        logger.info('Reading %d paragraphs...' % len(reader_examples))

        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(reader_examples) / 1e3))
        for batch in self._get_reader_loader(reader_examples, num_loaders): 
            if candidates or self.fixed_candidates:
                batch_cands = []
                for ex_id in batch[-1]:
                    batch_cands.append({
                        'input': s_tokens[ex_id[2]],
                        'cands': candidates[ex_id[0]] if candidates else None
                    })
                handle = self.reader.predict(
                    batch, batch_cands, async_pool=self.processes
                )
            else:
                handle = self.reader.predict(batch, async_pool=self.processes)
                # print(len(batch))
                # handle = self.reader.update(batch)
            result_handles.append((handle, batch[-1], batch[0].size(0)))

        # Iterate through the predictions, and maintain priority queues for
        # top scored answers for each question in the batch.
        queues = [[] for _ in range(len(queries))]
        for result, ex_id, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_id[i], s[i][0], e[i][0])
                    queue = queues[ex_id[i][0]]
                    if len(queue) < n_pars:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)

        # Arrange final top prediction data.
        all_predictions = []
        for q_idx, queue in enumerate(queues):
            predictions = []
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)
                # print('qidx {}, sidx {}: {}'.format(qidx, sidx, qs_to_score[qidx][sidx]))
                prediction = {
                    # 'doc_id': all_docids[qidx][rel_didx],
                    's_idx': sidx,
                    'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': float(score),
                    'sidx_score': float(qs_to_score[qidx][sidx])
                }
                '''
                if return_context:
                    prediction['context'] = {
                        'text': s_tokens[sidx].untokenize(),
                        'start': s_tokens[sidx].offsets()[s][0],
                        'end': s_tokens[sidx].offsets()[e][1],
                    }
                '''
                predictions.append(prediction)
            '''
            print('PRED', predictions[-1::-1])
            print('RANK', q_best[q_idx])
            print('ANS', answers[q_idx])
            print('QES', queries[q_idx])
            print()
            '''
            all_predictions.append(predictions[-1::-1])
        # sys.exit()

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        
        return paragraphs, all_predictions
