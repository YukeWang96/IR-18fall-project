import os
import gensim
import nltk
import pickle
from operator import itemgetter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.summarization import bm25

def docs_process(data_file, title_s=False, body_s=False):
    titles = []
    bodies = []
    counter = 0
    if os.path.isdir(data_file):
        dir = sorted(os.listdir(data_file))
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

    return titles, bodies, dir


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

def main(query_set, qc_dict, order_list, process_title, process_body):
    counter = 0
    scores_title = []
    scores_body = []
    qd_dict_list = list(qd_dict.values())

    for query_tokenized in query_set:

        qd_list = sorted(qd_dict_list[counter])
        qd_idx = [order_list.index(item) for item in qd_list]

        if process_title:
            doc_bm25_title = doc_process_bm25(list(itemgetter(*qd_idx)(titles)))
            average_idf_title = sum(map(lambda k: float(doc_bm25_title.idf[k]), doc_bm25_title.idf.keys())) / len(doc_bm25_title.idf.keys())
            tmp_score_title = doc_bm25_title.get_scores(query_tokenized, average_idf_title)
            scores_title.append(tmp_score_title)

        if process_body:
            doc_bm25_body = doc_process_bm25(list(itemgetter(*qd_idx)(docs)))
            average_idf_body = sum(map(lambda k: float(doc_bm25_body.idf[k]), doc_bm25_body.idf.keys())) / len(doc_bm25_body.idf.keys())
            tmp_score_body = doc_bm25_body.get_scores(query_tokenized, average_idf_body)
            scores_body.append(tmp_score_body)

        print("[" + str(counter) + "]: "  + str(query_tokenized))
        counter += 1

    return scores_title, scores_body

if __name__ == "__main__":

    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')

    with open("qd_dict.bin", "rb") as fqd:
        qd_dict = pickle.load(fqd)
    
    doc_dir = "docs_new_small"
    qry_file = "title-queries.301-450"

    fw_title = open("BM25_result_title.log", "w")
    fw_body = open("BM25_result_body.log", "w")

    process_title = True
    process_body = True

    qry_set = query_process(qry_file)
    titles, docs, order_list = docs_process(doc_dir, process_title, process_body)

    final_result_title, final_result_body = main(qry_set, qd_dict, order_list, process_title, process_body)

    if process_title and process_body:

        for qf in final_result_title:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_title.write(result)
        fw_title.close()
        
        print("BM_25 title finished....")

        for qf in final_result_body:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_body.write(result)
            
        fw_body.close()

        print("BM_25 body finished....")