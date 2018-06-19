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


lucene.initVM(maxheap='4096m')

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
hitEntities = topdocs.scoreDocs
hitDocs = topdocs.entityWeightedDocs
numTotalHits = topdocs.totalHits
numTotalDocs = topdocs.totalDocs

print("{} total matching entities ({} docs)".format(numTotalHits, numTotalDocs))

start = 0
end = min(numTotalHits, hitsPerPage)

# entities
for i in range(end):

    assert ScoreDocE.instance_(hitEntities[i])

    sde = hitEntities[i]
    sde = ScoreDocE.cast_(sde)
    entityDocs = searcher.search(
        TermQuery(Term("eid", str(sde.doc))), 1).scoreDocs
    if len(entityDocs) > 0:
        entityDoc = searcher.doc(entityDocs[0].doc)
        print("entityDoc name={}\ttype={}\tscore={:.3f}".
              format(entityDoc.get("name"), entityDoc.get("type"), sde.score))

    docNum = sde.docNum
    print("#paragraphs={}".format(docNum))

    # print(lucene.JArray('object').instance_(sde.docs, ScoreDoc))

    # # TODO is it python jcc's problem?
    # docs = lucene.JArray('object').cast_(sde.docs, ScoreDoc)
    # for sd in docs:
    #     print("doc={}\tscore={}".format(sd.doc, sd.score))

    print()

# entity score weighted docs
ewdoc_end = min(numTotalDocs, hitsPerPage)
for i in range(ewdoc_end):
    print(hitDocs[i])
