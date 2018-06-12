import json
import os
import spacy
import sqlite3
import time
import threading
import unicodedata
from datetime import datetime
from queue import Queue


# Ref.
# https://docs.python.org/3/library/queue.html
# https://www.ploggingdev.com/2017/01/multiprocessing-and-multithreading-in-python-3/
# https://spacy.io/usage/facts-figures#ner-accuracy-ontonotes5

# A copy of
# https://github.com/facebookresearch/DrQA/blob/master/drqa/retriever/doc_db.py
class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

        self.ner_col_name = 'p_ner_spacy'

        # list columns
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(documents)")
        cols = cursor.fetchall()
        print(cols)
        cursor.close()

        found_ner_col = False
        for col in cols:
            if self.ner_col_name == col[1]:
                found_ner_col = True
        if not found_ner_col:
            # add NER column to wiki_db
            cursor = self.connection.cursor()
            cursor.execute(
                "ALTER TABLE documents ADD {} TEXT".format(self.ner_col_name))
            cursor.close()
            print('Added a column: {}'.format(self.ner_col_name))

        try:
            # add index for speed
            cursor = self.connection.cursor()
            cursor.execute("CREATE INDEX idx ON documents(id)")
            cursor.close()
        except sqlite3.OperationalError as e:
            print(e)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_no_ner_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE {} IS NULL".
                       format(self.ner_col_name))
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_ner_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE {} IS NOT NULL".
                       format(self.ner_col_name))
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (unicodedata.normalize('NFD', doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_doc_p_ents(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT {} FROM documents WHERE id = ?".format(self.ner_col_name),
            (unicodedata.normalize('NFD', doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def update_ner_doc(self, doc_id, v):
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE documents SET {}=? WHERE id = ?".format(self.ner_col_name),
            (v, unicodedata.normalize('NFD', doc_id))
        )
        cursor.close()


def preprocess_worker(doc_id, log_interval=1000):
    global wiki_db
    global nlp_spacy
    global doc_count

    with threading_lock:
        doc_text = wiki_db.get_doc_text(doc_id)

    doc_paragraphs_json = get_doc_paragraphs_json(doc_text, nlp_spacy)

    with threading_lock:
        wiki_db.update_ner_doc(doc_id, doc_paragraphs_json)

    doc_count += 1
    if doc_count % log_interval == 0:
        with threading_lock:
            print(datetime.now(), doc_count)

            if doc_count % (3 * log_interval) == 0:
                wiki_db.connection.commit()


def get_doc_paragraphs_json(doc_text, nlp):
    paragraph_infos = list()
    paragraphs = doc_text.split('\n\n')
    for p_idx, p in enumerate(paragraphs):
        paragraph_infos.append(get_p_dict(p, nlp))
    return json.dumps({'paragraphs': paragraph_infos})


def get_p_dict(p, nlp):
    p_doc = nlp(p.strip())  # trim and nlp using spaCy

    # NER
    ents = list()
    for entity in p_doc.ents:
        ents.append({'text': entity.text,
                     'start_char': entity.start_char,
                     'end_char': entity.end_char,
                     'label_': entity.label_,
                     'label': entity.label})

    return {'text': p, 'ents': ents}


wiki_db = None
doc_ids = None
nlp_spacy = None
doc_count = 0

threading_lock = threading.Lock()


def main():
    global wiki_db
    global doc_ids
    global nlp_spacy

    n_threads = 8

    # https://spacy.io/usage/facts-figures#benchmarks-models-english
    # python3 -m spacy download en_core_web_lg
    nlp_spacy = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])

    start = time.time()

    def worker():
        while True:
            item = q.get(block=True, timeout=None)
            if item is None:
                break
            preprocess_worker(item)
            q.task_done()

    q = Queue()
    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    wiki_db = DocDB(
        db_path=os.path.join(os.path.expanduser('~'), 'common', 'wikipedia',
                             'docs.db'))
    print('Getting doc ids..')
    doc_ids = wiki_db.get_no_ner_doc_ids()
    print('# wiki docs', len(doc_ids))
    for doc_id in doc_ids:
        q.put(doc_id)

    # block until all tasks are done
    q.join()

    # stop workers
    print('Stopping..')
    for i in range(n_threads):
        q.put(None)
    for t in threads:
        t.join()

    wiki_db.connection.commit()
    print('DB commit')

    # print(threading.enumerate())

    print("Execution time = {0:.5f}".format(time.time() - start))

    wiki_db.close()


def label_no_label_ids():
    wikipedia_db = DocDB(
        db_path=os.path.join(os.path.expanduser('~'), 'common', 'wikipedia',
                             'docs.db'))
    print('Getting doc ids..')
    wikidoc_ids = wikipedia_db.get_doc_ids()
    print('# wiki docs', len(wikidoc_ids))

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])

    resurrection_count = 0

    for didx, doc_id in enumerate(wikidoc_ids):
        doc_p_ents = wikipedia_db.get_doc_p_ents(doc_id)

        found_invalid_doc = False

        if doc_p_ents:
            doc_dict = json.loads(doc_p_ents)

            paragraphs = doc_dict['paragraphs']
            for p_idx, p in enumerate(paragraphs):
                ents = p['ents']

                found_invalid_p = False
                for entity in ents:
                    if 'label' not in entity:
                        found_invalid_p = True
                        break

                if found_invalid_p:
                    found_invalid_doc = True
                    break
        else:
            found_invalid_doc = True

        if found_invalid_doc:
            wikipedia_db.update_ner_doc(doc_id,
                                        get_doc_paragraphs_json(
                                            wikipedia_db.get_doc_text(doc_id),
                                            nlp)
                                        )
            resurrection_count += 1

        if (didx + 1) % 100000 == 0:
            print(datetime.now(), didx + 1, resurrection_count, sep='\t')

    print("resurrection_count", resurrection_count)


if __name__ == '__main__':
    main()
    # label_no_label_ids()
