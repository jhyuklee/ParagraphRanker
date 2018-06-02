from datetime import datetime
import lucene
import os
import json
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, IntPoint
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
import sys
import time
import threading


# Ref.
# http://svn.apache.org/viewvc/lucene/pylucene/trunk/samples/IndexFiles.py


class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, store_dir, analyzer, docs_path):
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)

        store = SimpleFSDirectory(Paths.get(store_dir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(store, config)

        self.index_docs(docs_path)
        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        self.writer.commit()
        self.writer.close()
        ticker.tick = False
        print('done')

    def index_docs(self, docs_path):
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        num_paragraphs = 0
        num_empty_paragraphs = 0

        # TODO get docs from our sqlite db
        with open(docs_path, 'r', encoding='utf-8') as f:
            for d_idx, line in enumerate(f):
                doc_dict = json.loads(line)

                paragraphs = doc_dict['paragraphs']
                for p_idx, p in enumerate(paragraphs):

                    if len(p['text']) == 0:
                        num_empty_paragraphs += 1
                        continue

                    lucene_doc = Document()
                    lucene_doc.add(Field("wiki_doc_id", doc_dict['id'], t1))
                    lucene_doc.add(Field("p_idx", str(p_idx), t1))
                    lucene_doc.add(Field("content", p['text'], t2))

                    # Named-entities
                    ents = p['ents']
                    if len(ents) > 0:
                        for entity in ents:
                            lucene_doc.add(Field("entity", entity['text'], t2))
                            lucene_doc.add(Field("entity_label",
                                                 entity['label_'], t2))
                            lucene_doc.add(Field("entity_start",
                                                 str(entity['start_char']),
                                                 t1))
                            lucene_doc.add(Field("entity_end",
                                                 str(entity['end_char']),
                                                 t1))

                    self.writer.addDocument(lucene_doc)
                    num_paragraphs += 1

                if (d_idx + 1) % 10000 == 0:
                    print(datetime.now(), 'Added #docs', d_idx + 1,
                          '#paragraphs', num_paragraphs)

        print('#empty_paragraphs', num_empty_paragraphs)


class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


if __name__ == '__main__':
    lucene.initVM()
    print('lucene', lucene.VERSION)
    start = datetime.now()
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        IndexFiles(os.path.join(base_dir, 'eqa_index'),
                   StandardAnalyzer(), './data/wiki_spacy_ner.json')
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Failed: ", e)
        raise e
