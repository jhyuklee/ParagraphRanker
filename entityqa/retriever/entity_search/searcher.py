import lucene
import os

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


# git clone https://github.com/donghyeonk/entity-search.git

# Ref.
# https://github.com/apache/lucene-solr/blob/branch_6_5/lucene/demo/src/java/org/apache/lucene/demo/SearchFiles.java


lucene.initVM()

q = 'highest mountain'
field = 'content'
index_dir = os.path.join('/media/donghyeonkim/'
                         'f7c53837-2156-4793-b2b1-4b0578dffef1/entityqa',
                         'index')
hitsPerPage = 10

reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)))
searcher = IndexSearcherE(reader)

analyzer = StandardAnalyzer()
qparser = QueryParser(field, analyzer)
query = qparser.parse(q)

print("Searching for:", query.toString(field))

topdocs = searcher.searchE(query, 5 * hitsPerPage)
topdocs = TopDocsE.cast_(topdocs)
hits = topdocs.scoreDocs
numTotalHits = topdocs.totalHits
numTotalDocs = topdocs.totalDocs

print("{} total matching entities ({} docs)".format(numTotalHits, numTotalDocs))

start = 0
end = min(numTotalHits, hitsPerPage)

for i in range(end):
    sde = ScoreDocE.cast_(hits[i])
    entityDocs = searcher.search(
        TermQuery(Term("eid", str(sde.doc))), 1).scoreDocs
    if len(entityDocs) > 0:
        entityDoc = searcher.doc(entityDocs[0].doc)
        print("entityDoc name={}\ttype={}\tscore={:.3f}".
              format(entityDoc.get("name"), entityDoc.get("type"), sde.score))

    segNum = sde.segNum
    print("#paragraphs={}".format(segNum))

    # # TODO is it python jcc's problem?
    # segs = lucene.JArray('object').cast_(sde.segs, ScoreDoc)
    # for sd in segs:
    #     print("doc={}\tscore={}".format(sd.doc, sd.score))

    print()
