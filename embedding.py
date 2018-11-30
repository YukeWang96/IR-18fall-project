import re
import numpy as np
import gensim
import sys
import os
from random import randint

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

model = gensim.models.KeyedVectors.load_word2vec_format('/home/dheeraj/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
model.similarity(w1='clean',w2='neat')

#sys.exit()

def get_hist(query,body,flip_probability=0):
    body_sim=np.array([])
    for q in query:
        for sentence in body:
            wordList = re.sub("[^\w]", " ", sentence).split()
            sim = []
            for x in wordList:
                try:
                    do_flip = np.random.binomial(1,flip_probability)
                    if do_flip == 1:
                        similar_words = model.most_similar(positive=x)[:5]
                        cosine_score = np.array([y[1] for y in similar_words])
                        cosine_score = cosine_score/sum(cosine_score) # reweight the distribution
                        draw = np.random.multinomial(1,cosine_score)
                        index = np.where(draw == 1)[0][0]
                        new_word = similar_words[index][0]
                        sim.append(model.similarity(w1=q,w2=new_word))
                    else:
                        sim.append(model.similarity(w1=q,w2=x))
                except:
                    continue
            
            body_sim = np.append(body_sim,sim)
    
    body_hist = np.histogram(body_sim,range=(-1,1),bins=50)[0]
    body_hist = body_hist/sum(body_hist) # normalize histogram
    return body_hist


# removes all non-alphabets and convert to lower case
# and remove stop words
def strip_special_chars(line):
    u = line.strip()
    u = u.replace('-', ' ')
    u = re.sub(r'[^a-zA-Z ]+', '', u)
    u = u.lower()

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(u) # compare the results with the manual splitter
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    u = ' '.join(filtered_sentence)
    
    return u
    

def parse_queries():
    with open('test-data/title-queries.301-450') as f:
    #with open('test-data/sampleQ') as f:
        lines = f.readlines()
    queries = {} # returns qid:[list of query words]
    for line in lines:
        u = line.strip()
        u = u.lower()
        u = u.replace('-',' ')
        u = re.sub(r'[^0-9a-zA-Z ]+', '', u)
        qid = re.sub("[^0-9]", " ", u).split()[0]
        qwords = re.sub("[^a-z]", " ", u).split()

        stop_words = set(stopwords.words('english'))
        qwords = [w for w in qwords if not w in stop_words]
        assert len(qwords) > 0
        queries[qid] = qwords
    return queries
        
def get_qrels_map():
    with open('test-data/qrels.trec6-8.nocr') as f:
    #with open('test-data/sampleR') as f:
        lines = f.readlines()
    qrels = {}
    for line in lines:
        u = line.strip()
        qid = u.split()[0]
        rdoc = u.split()[2:]
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(rdoc)
    return qrels

def parse_doc(doc):
    lines = []
    with open('test-data/docs/'+doc,'r') as f:
    #with open('test-data/example/'+doc,'r') as f:
        lines = f.readlines()
    title_p = []
    body_p = []
    ctr = 0

    for line in lines:
        if line.find('Body:') >= 0:
            break;
        ctr = ctr + 1
        title_p.append(line)

    body_p.extend(lines[ctr:])

    title = []
    body = []

    for x  in title_p:
        u = x.replace('Title:','')
        u = strip_special_chars(u)
        if re.match(r'.*[a-z]+.*', u):
            title.append(u)

    for x  in body_p:
        u = x.replace('Body:','')
        u = strip_special_chars(u)
        if re.match(r'.*[a-z]+.*', u): # filter empty lines
            body.append(u)
            
    return title,body

def write_body_features(body_features,body_hist):
    ctr = 2
    for count in body_hist:
        body_features.write(str(ctr)+':'+str(count)+' ')
        ctr = ctr+1
    body_features.write('\n')
    
# raw_docs can be titles or bodies
def doc_process_tf_idf(raw_docs):
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_docs]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)
    return corpus, dictionary, tf_idf

def single_query_docs_tf_idf(query_doc, corpus, dictionary, tf_idf):
    # handle the incoming query
    # query_doc = [w.lower() for w in word_tokenize(query)]
    # print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    # print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # print(query_doc_tf_idf)
    sims = gensim.similarities.Similarity('.', tf_idf[corpus], num_features=len(dictionary))
    # print(sims)
    # print(type(sims))
    return sims[query_doc_tf_idf]


queries = parse_queries()
qrels_map = get_qrels_map()

title_features = open('test-data/features/qtitle.txt','w')
body_features = open('test-data/features/qbody.txt','w')

examples=0
title_set = {}
body_set = {}

dir = os.listdir('test-data/docs')
#dir = os.listdir('test-data/example')
for fname in dir:
    title,body = parse_doc(fname)
    title_set[fname] = title
    body_set[fname] = body

#build tf-idf values for words in title and body
arg_t = []
arg_b = []
idx = 0
index_map_t = {}
index_map_b = {}

print('start processing docs')
for x in title_set:
    index_map_t[x] = idx;
    idx = idx + 1
    if len(title_set[x]) == 0:
        arg_t.append("")
    else:
        arg_t.append(title_set[x][0])

idx = 0    
for x in body_set:
    index_map_b[x] = idx;
    idx = idx + 1
    if len(body_set[x]) == 0:
        arg_b.append("")
    else:
        arg_b.append(body_set[x][0])
    
corpus_t, dictionary_t, tf_idf_t = doc_process_tf_idf(arg_t)
corpus_b, dictionary_b, tf_idf_b = doc_process_tf_idf(arg_b)

print('end processing docs')

#qid: tfidf values for each dcument wrt to the query as numpy array
query_tf_idf_title = {}
query_tf_idf_body = {}

for qid in queries:
    # need to build the query tf-idf matrix q:doc1,doc2,...
    query_tf_idf_title[qid] = single_query_docs_tf_idf(queries[qid],corpus_t,dictionary_t,tf_idf_t)*10
    query_tf_idf_body[qid] = single_query_docs_tf_idf(queries[qid],corpus_b,dictionary_b,tf_idf_b)*10

print('begin augmentation')

for key in qrels_map:
    for doc in qrels_map[key]:
        docname = doc[0]
        relevancy = doc[1]
        #if os.path.isfile('test-data/example/'+docname) == False:
        if os.path.isfile('test-data/docs/'+docname) == False:
            continue
        examples = examples + 1

        title = title_set[docname]
        body = body_set[docname]
            
        query = queries[key]
        title_hist = get_hist(query,title)
        body_hist = get_hist(query,body)
        #print(title_hist)

        doc_idx_t = index_map_t[docname]
        title_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_title[qid][doc_idx_t])+' ')
        
        ctr = 2
        for count in title_hist:
            title_features.write(str(ctr)+':'+str(count)+' ')
            ctr = ctr+1
        title_features.write('\n')

        doc_idx_b = index_map_b[docname]
        body_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_body[qid][doc_idx_b])+' ')
        write_body_features(body_features,body_hist)

        #generate augmented features for body
        body_hist1 = get_hist(query,body,0.3)
        body_hist2 = get_hist(query,body,0.3)
        body_hist3 = get_hist(query,body,0.3)
        body_hist4 = get_hist(query,body,0.3)

        body_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_body[qid][doc_idx_b])+' ')
        write_body_features(body_features,body_hist1)

        body_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_body[qid][doc_idx_b])+' ')
        write_body_features(body_features,body_hist2)

        body_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_body[qid][doc_idx_b])+' ')
        write_body_features(body_features,body_hist3)

        body_features.write(relevancy+' qid:'+key+ ' 1:'+str(query_tf_idf_body[qid][doc_idx_b])+' ')
        write_body_features(body_features,body_hist4)
        
title_features.close()
body_features.close()

#IMPORTANT: remember to do a shuffling during training/ during generation?
print('produced features for '+str(examples)+' training points')

# average_idf, copy the title to body if body is empty

# ssh yuke@winnie.cs.ucsb.edu
# 19256700
