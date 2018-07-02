#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import os
import sys
import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from root.retriever import utils
from root import tokenizers

import lucene
from java.nio.file import Paths

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
# from org.apache.lucene.search import ScoreDoc
from org.apache.lucene.search import TermQuery
from org.apache.lucene.store import FSDirectory

from kr.ac.korea.dmis.search import IndexSearcherE
from kr.ac.korea.dmis.search import ScoreDocE
from kr.ac.korea.dmis.search import TopDocsE


logger = logging.getLogger(__name__)


class EntityRetriever(object):

    def __init__(self, index_dir=None):
        """
        Args:
            index_dir: path to saved index
        """
        # Load from disk
        logger.info('Loading retriever %s' % index_dir)
        lucene.initVM(maxheap='8192m')

        reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
        self.searcher = IndexSearcherE(reader)
        self.analyzer = StandardAnalyzer()
        self.qparser = QueryParser('content', self.analyzer)
        self.digit_entities = ['ORDINAL', 'CARDINAL', 'PERCENT', 'DATE', 
                               'TIME', 'MONEY', 'QUANTITY', 'GPE', 'NORP']

    def closest_pars(self, query, k):
        # Search using searcher
        query = self.qparser.parse(query)
        topdocs = self.searcher.searchE(query, k)
        topdocs = TopDocsE.cast_(topdocs)
        hitEntities = topdocs.scoreDocs
        hitDocs = topdocs.entityWeightedDocs
        numTotalHits = topdocs.totalHits
        numTotalDocs = topdocs.totalDocs
        entities = []

        # Iterate each entities
        for i in range(min(numTotalHits, k)):
            assert ScoreDocE.instance_(hitEntities[i])

            sde = hitEntities[i]
            sde = ScoreDocE.cast_(sde)
            entityDocs = self.searcher.search(
                TermQuery(Term("eid", str(sde.doc))), 1).scoreDocs

            if len(entityDocs) > 0:
                entityDoc = self.searcher.doc(entityDocs[0].doc)
                if entityDoc.get("type") not in self.digit_entities:
                    entities.append({
                        'name': entityDoc.get('name'),
                        'type': entityDoc.get('type'),
                        'score': sde.score,
                    })
                    docNum = sde.docNum

        return hitDocs, entities

    def batch_closest_pars(self, queries, k, num_workers=None):
        # print('processing query', queries[0])
        '''
        with ThreadPool(num_workers) as threads:
            closest_pars = partial(self.closest_pars, k)
            results = threads.map(closest_pars, queries)
        '''
        
        pars = []
        ents = []
        for query in queries:
            par, ent = self.closest_pars(query, k)
            pars += [par]
            ents += [ent]
        return pars , ents
    
    def get_texts(self, pars):
        par_texts = []
        q2pid = []
        for i in range(len(pars)):
            text = []
            for j in range(len(pars[i])):
                text += [retriever.searcher.doc(pars[i][j].doc).get('content')]
            par_texts += text
            # q2pid += 

        return par_texts


if __name__ == '__main__':
    index_dir = os.path.join(os.path.expanduser('~'),
                             'github/entityqa/data/index')
    retriever = EntityRetriever(index_dir)
    pars, ents = retriever.batch_closest_pars(['mountain', 'korea'], 100)
    print(pars)
    print()
    print(ents)
    print()
    real_doc = retriever.get_texts(pars)
    # print(real_doc)
    print('Number of pars: {}'.format(len(real_doc)))
    print(real_doc[0])

