import os
import gensim
import nltk
import pickle
from operator import itemgetter
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def single_query_docs_tf_idf(query, corpus, dictionary, tf_idf):
    query_doc_bow = dictionary.doc2bow(query)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    sims = gensim.similarities.Similarity('.', tf_idf[corpus], num_features=len(dictionary))
    return sims[query_doc_tf_idf]


def docs_process(data_file, title_s, body_s):
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

            counter +=1
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
def doc_process_tf_idf(raw_docs):
    gen_docs = [[w.lower() for w in tokenizer.tokenize(text) if not w in stop_words] for text in raw_docs]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)
    return corpus, dictionary, tf_idf

def main(query_set, qd_dict, order_list, titles, docs, generate_title=True, generate_body=False):
    result_set_title = []
    result_set_body = []
    counter = 0
    qd_dict_list = list(qd_dict.values())

    for iter in range(len(query_set)):

        qd_list = sorted(qd_dict_list[iter])
        qd_idx = [order_list.index(item) for item in qd_list]
 
        if generate_title:
            corpus_title, dictionary_title, tf_idf_title = doc_process_tf_idf(list(itemgetter(*qd_idx)(titles)))
            result_title = single_query_docs_tf_idf(query_set[iter], corpus_title, dictionary_title, tf_idf_title)
            result_set_title.append(result_title)

        if generate_body:
            corpus_body, dictionary_body, tf_idf_body = doc_process_tf_idf(list(itemgetter(*qd_idx)(docs)))
            result_body = single_query_docs_tf_idf(query_set[iter], corpus_body, dictionary_body, tf_idf_body)
            result_set_body.append(result_body)

        print("[" + str(counter) + "]: "  + str(query_set[iter]))
        counter += 1

    return result_set_title, result_set_body


if __name__ == "__main__":

    tokenizer = RegexpTokenizer(r'\w+')
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    generate_title = True
    generate_body = True

    recover = False

    with open("qd_dict.bin", "rb") as fqd:
        qd_dict = pickle.load(fqd)
    
    if generate_title:
        fw_title = open("TF-IDF_result_title.log", "w")

    if generate_body:
        fw_body = open("TF-IDF_result_body.log", "w")

    doc_dir = "docs_new_small" # test_small
    qry_file = "title-queries.301-450"
    
    qry_set = query_process(qry_file)

    if recover:
        f_docs = open("docs_dump", "rb")
        docs = pickle.load(f_docs)

        f_titles = open("titles_dump", "rb")
        titles = pickle.load(f_titles)

        f_order_list = open("order_list_dump", "rb")
        order_list = pickle.load(f_order_list)
    else:
        if generate_title and generate_body:
            titles, docs, order_list = docs_process(doc_dir, title_s=generate_title, body_s=generate_body)

        if generate_title and not generate_body:
            titles, _, order_list = docs_process(doc_dir, title_s=generate_title, body_s=generate_body)
        
        if generate_body and not generate_title:
            _, docs, order_list = docs_process(doc_dir, title_s=generate_title, body_s=generate_body)

        # backup docs during readin process
        if generate_body:
            f_docs = open("docs_dump", "wb")
            pickle.dump(docs, f_docs)
            f_docs.close()
        
        if generate_title:
            f_titles = open("titles_dump", "wb")
            pickle.dump(titles, f_titles)
            f_titles.close()

        f_order_list = open("order_list_dump", "wb")
        pickle.dump(order_list, f_order_list)
        f_order_list.close()
    
    if generate_title and generate_body:
        final_result_title, final_result_body = main(qry_set, qd_dict, order_list, titles, docs, generate_title=True, generate_body=True)

        for qf in final_result_title:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_title.write(result)
        fw_title.close()
        print("[1/2] title part finished")

        for qf in final_result_body:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_body.write(result)
        fw_body.close()
        print("[2/2] body part finished")

    if generate_title and not generate_body:
        final_result_title, _ = main(qry_set, qd_dict, order_list, titles, docs, generate_title=True, generate_body=False)

        for qf in final_result_title:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_body.write(result)
        fw_body.close()
        print("[1/1] title part finished")

    if generate_body and not generate_title:
        _ , final_result_body = main(qry_set, qd_dict, order_list, titles, docs, generate_title=False, generate_body=True)

        for qf in final_result_body:
            result = ""
            for i in range(len(qf)):
                result += str(qf[i]) + " "
            result += "\n"
            fw_body.write(result)
        fw_body.close()
        print("[1/1] body part finished")
