import os
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from gensim.summarization import bm25

def docs_process(data_file, title_s=False, body_s=False):
    titles = []
    bodies = []
    counter = 0
    if os.path.isdir(data_file):
        dir = os.listdir(data_file)
        for fname in dir:
            fp = open(data_file + "/" + fname, "r")
            tmp_doc = fp.read().split("Body:")

            if title_s:
                doc_title = tmp_doc[0].replace("Title:", "").replace("\n", "")
                titles.append(doc_title)

            if len(tmp_doc) >= 2:
                if body_s:
                    doc_body = tmp_doc[1].replace("\n", "")
                    bodies.append(doc_body)
            else:
                if body_s:
                    doc_body = tmp_doc[0].replace("Title:", "").replace("\n", "")
                    bodies.append(doc_body)

            counter += 1
            print(counter)
    else:
        fp = open(data_file, "r")
        tmp_doc = fp.read().split("Body:")

        if title_s:
            doc_title = tmp_doc[0].replace("Title:", "").replace("\n", "")
            titles.append(doc_title)

        if len(tmp_doc) >= 2:
            if body_s:
                doc_body = tmp_doc[1].replace("\n", "")
                bodies.append(doc_body)
        else:
            if body_s:
                doc_body = tmp_doc[0].replace("Title:", "").replace("\n", "")
                bodies.append(doc_body)

    return titles, bodies


def query_process(file_dir):
    fp = open(file_dir, "r")
    query_set = []
    for line in fp:
        tmp_line = line.split(":")[1].replace("\n", '')
        token_tmp = tokenizer.tokenize(tmp_line)
        tmp = [w for w in token_tmp if not w in stop_words]
        query_set.append(tmp)
    return query_set


# raw_docs can be titles or bodies
def doc_process_bm25(raw_docs):
    gen_docs = [[w.lower() for w in tokenizer.tokenize(text) if not w in stop_words] for text in raw_docs]
    # dictionary = gensim.corpora.Dictionary(gen_docs)
    # corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    bm25Model = bm25.BM25(gen_docs)
    return bm25Model

def main(query_set, doc_bm25, average_idf):
    counter = 0
    scores = []
    for query_tokenized in query_set:
        tmp_score = doc_bm25.get_scores(query_tokenized, average_idf)
        scores.append(tmp_score)
        print("[" + str(counter) + "]: "  + str(query_tokenized))
        counter += 1
    return scores

if __name__ == "__main__":

    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    
    doc_dir = "docs"
    qry_file = "title-queries.301-450"

    fw = open("BM25_result.log", "w")
    search_in_title = True
    search_in_body = False

    qry_set = query_process(qry_file)
    titles, docs = docs_process(doc_dir, True, True)

    if search_in_title:
        doc_bm25 = doc_process_bm25(titles)
    else:
        doc_bm25 = doc_process_bm25(docs)

    average_idf = sum(map(lambda k: float(doc_bm25.idf[k]), doc_bm25.idf.keys())) / len(doc_bm25.idf.keys())
    final_result = main(qry_set, doc_bm25, average_idf)

    for qf in final_result:
        result = ""
        for i in range(len(qf)):
            result += str(qf[i]) + " "
        result += "\n"
        fw.write(result)
    
    fw.close()