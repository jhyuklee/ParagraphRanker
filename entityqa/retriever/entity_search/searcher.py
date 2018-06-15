import lucene
import os

from java.nio.file import Paths

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory

from kr.ac.korea.dmis.search import TopEntitiesCollector

# Ref.
# https://github.com/apache/lucene-solr/blob/branch_6_5/lucene/demo/src/java/org/apache/lucene/demo/SearchFiles.java


lucene.initVM()

# tec = TopEntitiesCollector(10, True)  # raise an error

q = 'highest mountain'
field = 'content'
index_dir = os.path.join('/media/donghyeonkim/'
                         'f7c53837-2156-4793-b2b1-4b0578dffef1/entityqa',
                         'index')
hitsPerPage = 10

reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
searcher = IndexSearcher(reader)

analyzer = StandardAnalyzer()
qparser = QueryParser(field, analyzer)
query = qparser.parse(q)

print("Searching for:", query.toString(field))

topdocs = searcher.search(query, 5 * hitsPerPage)
hits = topdocs.scoreDocs
numTotalHits = topdocs.totalHits

print(numTotalHits, "total matching documents")

start = 0
end = min(numTotalHits, hitsPerPage)

for i in range(end):
    doc = searcher.doc(hits[i].doc)
    wiki_doc_id = doc.get("wiki_doc_id")
    p_idx = doc.get("p_idx")
    content = doc.get("content")
    print()
    print(i, "Wiki doc id   : " + wiki_doc_id)
    print(i, "Paragraph idx : " + p_idx)
    print(i, "Paragraph text: " + content)


# TODO
