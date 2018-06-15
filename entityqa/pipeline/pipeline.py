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
from root.retriever import utils as r_utils
from root.retriever.tfidf_doc_ranker import TfidfDocRanker
from root.retriever.doc_db import DocDB
from root.retriever.eval import init, get_answer_label, fetch_text, tokenize_text 

from root.ranker.model import ParagraphRanker
from root.ranker.data import RankerDataset, RankerBatchSampler
from root.ranker.vector import ranker_train_batchify, ranker_dev_batchify

from root.reader import utils
from root.reader.model import DocReader
from root.reader.data import ReaderDataset, ReaderBatchSampler
from root.reader.vector import reader_batchify


logger = logging.getLogger(__name__)


class QAPipeline(object):
    GROUP_LENGTH = 0
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, retriever_path, db_path, ranker_path, reader_path, 
                 module_batch_size, module_max_loaders, module_cuda, strict=True,
                 fixed_candidates=None):
        """
        Args:
            strict: fail on empty queries or continue (and return empty result)
        """
        # Manually set
        self.max_loaders = module_max_loaders
        self.cuda = module_cuda
        self.batch_size = module_batch_size
        self.fixed_candidates = fixed_candidates

        # Load retriever
        self.retriever = TfidfDocRanker(retriever_path) 

        # Load pretrained ranker
        self.ranker = ParagraphRanker.load(ranker_path)
        if self.cuda:
            self.ranker.cuda()
        self.ranker.init_optimizer()

        # Load pretrained reader
        self.reader = DocReader.load(reader_path, normalize=False)
        if self.cuda:
            self.reader.cuda()
        self.reader.init_optimizer()

        logger.info('Initializing tokenizers...')
        tok_class = tokenizers.SimpleTokenizer
        annotators = tokenizers.get_annotators_for_model(self.ranker)
        tok_opts = {'annotators': annotators}
        db_config = {'options': {'db_path': db_path}}
        db_class = db_config.get('class', DocDB)
        db_opts = db_config.get('options', {})

        self.num_workers = 5
        self.processes = ProcessPool(
            self.num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, self.fixed_candidates)
        )

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

    def _get_ranker_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = RankerDataset(data, self.ranker,
                                neg_size=1, 
                                allowed_size=1000)
        sampler = RankerBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=ranker_dev_batchify,
            pin_memory=self.cuda,
        )

        return loader

    def _get_reader_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = ReaderBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=reader_batchify,
            pin_memory=self.cuda,
        )

        return loader

    def get_documents(self, queries, n_docs):
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.retriever.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.retriever.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)
        
        # Flatten document ids and retrieve text from database.
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
        
        return q_tokens, s_tokens, flat_splits, all_docids, all_doc_scores, \
            did2didx, didx2sidx

    def get_paragraphs_entities(self, queries, n_pars):
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d pars...' % n_pars)

        ranked = self.retriever.batch_closest_docs(
            queries, k=n_pars, num_workers=self.num_workers
        )
        all_parids, all_parmeta = zip(*ranked)
        
        # TODO
        # Fetch text of paragraphs based on par ids

        # Fetch text of queries
        q_tokens = self.processes.map_async(tokenize_text, queries)
        q_tokens = q_tokens.get()
        
        return q_tokens, p_tokens, p_raws, all_parids, all_parmeta 

    def update(self, queries, answers, candidates=None, n_docs=20, n_pars=200):
        t0 = time.time()
        q_tokens, s_tokens, flat_splits, all_docids, all_doc_scores, did2didx, \
            didx2sidx = self.get_documents(queries, n_docs)

        paragraphs = []
        for qidx in range(len(queries)):
            rel_pars = []
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    rel_pars += [flat_splits[sidx]]
            paragraphs += [rel_pars]

        answers_pars = zip(answers, paragraphs)
        get_al_partial = partial(get_answer_label, match='regex', use_text=True)
        answer_labels = self.processes.map(get_al_partial, answers_pars)
        # print(queries[2])
        # print(answers[2])
        # print([(p, a) for p, a in zip(paragraphs[2][:200], answer_labels[2][:200]) if a == 1.0])
        # TODO: Not all paragraphs that contain answers give clue on the answers
        assert len(paragraphs) == len(answer_labels)
        assert len(paragraphs[0]) == len(answer_labels[0])

        examples = []
        checksum = 0
        for qidx in range(len(queries)):
            p_idx = 0
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(q_tokens[qidx].words()) > 0 and
                            len(s_tokens[sidx].words()) > 0):
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': q_tokens[qidx].words(),
                            'document': s_tokens[sidx].words(),
                            'target': answer_labels[qidx][p_idx],
                        })
                        p_idx += 1
                    else:
                        assert False, 'Should not be zero'
            checksum += p_idx

        # Docids overlap between queries
        logger.info('Ranking {}/{} paragraphs...'.format(len(examples), checksum))

        # Push all examples through the document encoder.
        # We decode argmax start/end indices asychronously on CPU.
        ex_ids = []
        train_loss = utils.AverageMeter()
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_ranker_loader(examples, num_loaders):
            train_loss.update(*self.ranker.update(batch))

        logger.info('ranker loss = {:.3f}'.format(train_loss.avg))
        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        
        return train_loss.avg

    def predict(self, queries, candidates=None, n_docs=20, n_pars=200):
        t0 = time.time()
        q_tokens, s_tokens, flat_splits, all_docids, all_doc_scores, did2didx, \
            didx2sidx = self.get_documents(queries, n_docs)

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
                            'document': s_tokens[sidx].words(),
                        })
                    else:
                        assert False, 'Should not be zero'

        logger.info('Ranking %d paragraphs...' % len(examples))

        # Push all examples through the document encoder.
        # We decode argmax start/end indices asychronously on CPU.
        ex_ids = []
        sims = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_ranker_loader(examples, num_loaders):
            par_encoded, q_encoded = self.ranker.encode(batch)
            sim = torch.sum(torch.mul(par_encoded, q_encoded), dim=1)
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
            par_sim = sim.item()

            if qidx not in q_gather:
                q_gather[qidx] = []
            q_gather[qidx] += [[sidx, par_sim, rel_didx, doc_sim, doc_sim * par_sim]]

            # Search score by q and s
            if qidx not in qs_to_score:
                qs_to_score[qidx] = {}
            qs_to_score[qidx][sidx] = sim.item()

        # Filter only top ranked paragraphs
        n_paragraphs = n_pars
        q_best = {}
        for qidx, pairs in q_gather.items():
            pairs.sort(key=operator.itemgetter(1), reverse=True)
            q_best[qidx] = pairs[:n_paragraphs]

        # Untokenize paragraphs
        paragraphs = []
        for qidx in range(len(queries)):
            paragraphs += [[flat_splits[k[0]] for k in q_best[qidx]]]

        # Ready for reader inputs
        reader_examples = []
        for qidx in range(len(queries)):
            for sidx, _, rel_didx, _, _ in q_best[qidx]:
                if (len(q_tokens[qidx].words()) > 0 and
                        len(s_tokens[sidx].words()) > 0):
                    reader_examples.append({
                        'id': (qidx, rel_didx, sidx),
                        'question': q_tokens[qidx].words(),
                        'document': s_tokens[sidx].words(),
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
                prediction = {
                    'doc_id': all_docids[qidx][rel_didx],
                    's_idx': sidx,
                    'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': float(score),
                    'sidx_score': float(qs_to_score[qidx][sidx])
                }
                predictions.append(prediction)
            all_predictions.append(predictions[-1::-1])

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        
        return paragraphs, all_predictions
