# coding = "UTF-8"
import os
import pickle
from operator import itemgetter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def query_process(file_dir):
    fp = open(file_dir, "r")
    query_set = []
    for line in fp:
        tmp_line = line.split(":")[1].replace("\n", '')
        token_tmp = tokenizer.tokenize(tmp_line)
        tmp = [w for w in token_tmp if not w in stop_words]
        query_set.append(tmp)
    return query_set


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
                gen_docs = [w.lower() for w in tokenizer.tokenize(doc_title) if not w in stop_words]
                titles.append(gen_docs)
                
            if len(tmp_doc) >= 2:
                if body_s:
                    doc_body = tmp_doc[1].replace("\n", "")
                    gen_docs = [w.lower() for w in tokenizer.tokenize(doc_body) if not w in stop_words]
                    bodies.append(gen_docs)
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

def get_min_interval_between_list(temp_list_1, temp_list_2):
    
    minimum = 99999

    for it_1 in temp_list_1:
        for it_2 in temp_list_2:
            if it_1 - it_2  > 0:
                temp = it_1 - it_2
            else:
                temp = it_2 - it_1
            
            if temp < minimum:
                minimum = temp
    
    return minimum

def get_min_interval(w_loc):

    keys = [k for k in w_loc.keys()]
    global_minimum = 99999
    
    for i in range(0, len(keys) - 1):
        temp_1 = w_loc[keys[i]]
        for j in range(i+1, len(keys)):
            temp_2 = w_loc[keys[j]]
            temp_minimum = get_min_interval_between_list(temp_1, temp_2)

            if temp_minimum < global_minimum:
                global_minimum = temp_minimum
        
    return global_minimum


def distance(tokenized_query, tokenized_docs_sets):

    min_dist = []

    tk_qry_st = set(tokenized_query)

    for doc in tokenized_docs_sets:

        dict_qry = {}
        for i in range(len(doc)):
            if doc[i] in tk_qry_st:
                if doc[i] not in dict_qry.keys():
                    dict_qry[doc[i]] = []
                dict_qry[doc[i]].append(i)

        min_interval = get_min_interval(dict_qry)
        min_dist.append(min_interval)

    return min_dist


def main(qry_set, doc_set, qd_dir, order_list):

    counter = 0
    qd_dict_list = list(qd_dict.values())
    min_dist_global = []

    for qr in qry_set:

        qd_list = sorted(qd_dict_list[counter])
        qd_idx = [order_list.index(item) for item in qd_list]

        tmp_min = distance(qr, list(itemgetter(*qd_idx)(doc_set)))
        min_dist_global.append(tmp_min)
        
        print("[" + str(counter) + "] " + str(qr)) 
        counter += 1
    
    return min_dist_global

if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')

    fw_min = open("min_dist.log", "w")

    doc_dir = "docs_new_small"  
    qry_file = "title-queries.301-450"

    qry_set = query_process(qry_file)
    _, docs, order_list = docs_process(doc_dir, title_s=False, body_s=True)

    with open("qd_dict.bin", "rb") as fqd:
        qd_dict = pickle.load(fqd)

    min_dist_global = []

    min_dist_global = main(qry_set, docs, qd_dict, order_list)

    # total_line = len(min_dist_global)
    # total_width = len(min_dist_global[0])

    for iter in range(len(min_dist_global)):
        result = ""
        for i in range(len(min_dist_global[iter])):
            result += str(min_dist_global[iter][i]) + " "
        result += "\n"
        fw_min.write(result)
