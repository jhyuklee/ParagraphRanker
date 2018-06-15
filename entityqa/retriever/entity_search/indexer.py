from datetime import datetime
import lucene
import os
import json
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, IntPoint, \
    BinaryDocValuesField, StoredField
from org.apache.lucene.index import \
    DocValuesType, FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import BytesRef
import sys
import time
import threading
from preprocess_nlp import DocDB
from utils import write_vint

# Ref.
# http://svn.apache.org/viewvc/lucene/pylucene/trunk/samples/IndexFiles.py
# http://lucene.apache.org/core/6_5_0/core/index.html


class Indexer(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, store_dir, analyzer, db_path):
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)

        store = SimpleFSDirectory(Paths.get(store_dir))
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(store, config)

        # TODO checksum
        self.wiki_db = DocDB(db_path=db_path)

        print('Getting docs..', db_path)
        self.doc_ids = self.wiki_db.get_ner_doc_ids()
        print('# wiki docs', len(self.doc_ids))
        assert len(self.doc_ids) == 5075182

        self.entity2idx = dict()
        self.idx2entity = dict()
        self.entity2idx['UNK'] = 0
        self.idx2entity[0] = 'UNK'
        self.entitytype2idx = dict()
        self.entitytype2idx['UNK'] = 0
        self.entity_dict = dict()
        print('Init. Done')

    def index_docs(self, log_interval=100000):
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS)

        t2_tk = FieldType()
        t2_tk.setStored(True)
        t2_tk.setTokenized(True)
        t2_tk.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        bin_dv_ft = FieldType()
        bin_dv_ft.setDocValuesType(DocValuesType.BINARY)
        bin_dv_ft.setStored(True)
        bin_dv_ft.setIndexOptions(IndexOptions.DOCS)

        num_paragraphs = 0
        num_empty_paragraphs = 0

        # get docs from our sqlite db
        for d_idx, doc_id in enumerate(self.doc_ids):
            doc_p_ents = self.wiki_db.get_doc_p_ents(doc_id)

            assert doc_p_ents

            doc_dict = json.loads(doc_p_ents)

            paragraphs = doc_dict['paragraphs']
            for p_idx, p in enumerate(paragraphs):

                # TODO remove .strip() after docs.db splitter change
                p_text = p['text'].strip()

                if len(p_text) == 0:
                    num_empty_paragraphs += 1
                    continue

                lucene_doc = Document()
                lucene_doc.add(Field("wiki_doc_id", doc_id, t2_tk))  # TODO
                lucene_doc.add(Field("p_idx", str(p_idx), t1))
                lucene_doc.add(Field("content", p_text, t2_tk))

                # Named-entities
                ents = p['ents']
                ent_set = set()
                if len(ents) > 0:

                    entity_idx_set = set()
                    entity_type_id_set = set()
                    entity_positions = list()

                    for entity in ents:

                        assert 'label' in entity, 'doc_id={}'.format(doc_id)

                        entity_key = entity['text'] + '\t' + entity['label_']

                        etypeidx = self.entitytype2idx.get(entity['label'])
                        if etypeidx is None:
                            etypeidx = len(self.entitytype2idx)
                            self.entitytype2idx[entity['label']] = etypeidx

                        eidx = self.entity2idx.get(entity_key)
                        if eidx is None:
                            eidx = len(self.entity2idx)
                            self.entity2idx[entity_key] = eidx
                            self.idx2entity[eidx] = entity_key
                            self.entity_dict[eidx] = \
                                (entity['text'], entity['label_'], etypeidx)

                        entity_idx_set.add(eidx)
                        entity_type_id_set.add(etypeidx)
                        entity_positions.append((eidx, entity['start_char'],
                                                 entity['end_char']))

                        ent_set.add((eidx, etypeidx))

                    if len(entity_idx_set) > 0:
                        lucene_doc.add(
                            Field("entity_id",
                                  '\t'.join([str(eidx)
                                             for eidx in entity_idx_set]),
                                  t2_tk))
                        lucene_doc.add(
                            Field("entity_type_id",
                                  '\t'.join([str(etid)
                                             for etid in entity_type_id_set]),
                                  t2_tk))
                        lucene_doc.add(
                            Field("entity_position",
                                  '\t'.join(['{},{},{}'.
                                            format(eidx, start_char, end_char)
                                             for eidx, start_char, end_char
                                             in entity_positions]),
                                  t1))

                lucene_doc.add(
                    BinaryDocValuesField("eqa_bin",
                                         BytesRef(get_binary4dvs(ent_set))))
                # # debug
                # lucene_doc.add(
                #     StoredField("eqa_bin_store",
                #                 BytesRef(get_binary4dvs(ent_set))))

                self.writer.addDocument(lucene_doc)
                num_paragraphs += 1

                if num_paragraphs % log_interval == 0:
                    print(datetime.now(), 'Added #paragraphs', num_paragraphs,
                          '#wikidocs', d_idx + 1,
                          '#entities', len(self.entity_dict))

        print('#paragraphs', num_paragraphs)
        print('#skipped_empty_paragraphs', num_empty_paragraphs)

        print('\nAdding entity docs..')
        for e_dict_idx, entity_idx in enumerate(self.entity_dict):
            # skip UNK
            if entity_idx == self.entity2idx['UNK']:
                continue
            ename, etype, etype_idx = self.entity_dict[entity_idx]
            entity_doc = Document()
            entity_doc.add(Field("name", ename, t2_tk))
            entity_doc.add(Field("type", etype, t1))
            entity_doc.add(Field("eid", str(entity_idx), t1))
            entity_doc.add(Field("etid", str(etype_idx), t1))
            self.writer.addDocument(entity_doc)
            if (e_dict_idx + 1) % (10 * log_interval) == 0:
                print(datetime.now(), '#entities', e_dict_idx + 1)

        print('#entities', len(self.entity2idx) - 1)

        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        self.writer.commit()
        self.writer.close()
        ticker.tick = False
        print('done')


def get_binary4dvs(ent_set):
    binary = bytes()
    ent_size = len(ent_set)
    if ent_size == 0:
        return binary
    elif ent_size == 1:  # single -> even
        for eidx, etype in ent_set:
            assert eidx >= 0 and etype >= 0
            binary += write_vint(eidx << 1)
            binary += write_vint(etype)
            break  # double check
    else:  # multiple -> odd
        binary += write_vint((ent_size << 1) + 1)  # write # of entities
        for eidx, etype in ent_set:
            assert eidx >= 0 and etype >= 0
            binary += write_vint(eidx)
            binary += write_vint(etype)
    return binary


class Ticker(object):
    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


if __name__ == '__main__':
    # base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    output_dir = os.path.join('/media/donghyeonkim/'
                              'f7c53837-2156-4793-b2b1-4b0578dffef1/entityqa',
                              'index')
    db_filepath = os.path.join(os.path.expanduser('~'), 'common',
                               'wikipedia', 'docs.db')

    lucene.initVM()
    print('lucene', lucene.VERSION)
    start = datetime.now()
    try:
        indexer = Indexer(output_dir, StandardAnalyzer(), db_filepath)
        indexer.index_docs()
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Failed: ", e)
        raise e
