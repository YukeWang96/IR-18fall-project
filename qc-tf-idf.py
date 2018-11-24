import os
import gensim
from nltk.tokenize import word_tokenize

# print(dir(gensim))

def single_query_docs_tf_idf(query, corpus, dictionary, tf_idf):
    # handle the incoming query
    query_doc = [w.lower() for w in word_tokenize(query)]
    # print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # print(query_doc_tf_idf)
    sims = gensim.similarities.Similarity('.', tf_idf[corpus], num_features=len(dictionary))
    # print(sims)
    # print(type(sims))
    return sims[query_doc_tf_idf]


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
            
            if len(tmp_doc) > 2:
                if body_s:
                    doc_body = tmp_doc[1].replace("\n", "")
                    bodies.append(doc_body)

            counter +=1
            print(counter)
    else:
        fp = open(data_file, "r")
        tmp_doc = fp.read().split("Body:")

        if title_s:
                doc_title = tmp_doc[0].replace("Title:", "").replace("\n", "")
                titles.append(doc_title)

        if len(tmp_doc) > 2:    
            if body_s:
                    doc_body = tmp_doc[1].replace("\n", "")
                    bodies.append(doc_body)

    return titles, bodies


def query_process(file_dir):
    fp = open(file_dir, "r")
    query_set = []
    for line in fp:
        tmp = line.split(":")[1].replace("\n", '')
        query_set.append(tmp)
    return query_set


# raw_docs can be titles or bodies
def doc_process_tf_idf(raw_docs):
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_docs]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)
    return corpus, dictionary, tf_idf

def main(query_set, corpus, dictionary, tf_idf):
    result_set=[]
    for qry in query_set:
        result = single_query_docs_tf_idf(qry, corpus, dictionary, tf_idf)
        result_set = result
    return result_set

if __name__ == "__main__":
    # raw_documents = [
    #     "I'm taking the show on the road.",
    #     "My socks are a force multiplier.",
    #     "I am the barber who cuts everyone's hair who doesn't cut their own.",
    #     "Legend has it that the mind is a mad monkey.",
    #     "I make my own fun."
    # ]
    #
    search_in_title = True
    search_in_body = False

    doc_dir = "docs"
    qry_file = "title-queries.301-450"

    query = "krone devaluation hits carlsberg"
    titles, docs = docs_process(doc_dir, True, True)
    
    qry_set = query_process(qry_file)

    # print(qry_set)
    # print(titles)
    # print(docs)

    if search_in_title:
        corpus, dictionary, tf_idf = doc_process_tf_idf(titles)
    else:
        corpus, dictionary, tf_idf = doc_process_tf_idf(docs)
    
    main(qry_set, corpus, dictionary, tf_idf)