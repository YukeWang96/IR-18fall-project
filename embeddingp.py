# mult-threadead script to do data augmentation and histogram generation
# author: dheeraj baby

import re
import numpy as np
import gensim
import sys
import os
import pickle
from random import randint,choice

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from threading import Thread

import warnings
warnings.filterwarnings("ignore")


#sys.exit()

def get_syn(word):
    syns = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            syns.append(l.name())
    if len(syns) < 10:
        return syns
    else:
        return syns[:10]

def get_hist(query,body,flip_probability=0):
    body_sim=np.array([])
    global model
    for q in query:
        for sentence in body:
            wordList = re.sub("[^\w]", " ", sentence).split()
            sim = []
            for x in wordList:
                try:
                    do_flip = np.random.binomial(1,flip_probability)
                    if do_flip == 1:
                        similar_words = get_syn(x)
                        new_word = choice(similar_words)
                        sim.append(model.similarity(w1=q,w2=new_word))
                    else:
                        sim.append(model.similarity(w1=q,w2=x))
                except:
                    continue
            
            body_sim = np.append(body_sim,sim)
    
    body_hist = np.histogram(body_sim,range=(-1,1),bins=50)[0]
    body_hist = body_hist/sum(body_hist) # normalize histogram
    return body_hist

stop_words = set(stopwords.words('english'))
# removes all non-alphabets and convert to lower case
# and remove stop words
def strip_special_chars(line):
    global stop_words
    u = line.strip()
    u = u.replace('-', ' ')
    u = re.sub(r'[^a-zA-Z ]+', '', u)
    u = u.lower()

    
    ## below line is really really slow
#    word_tokens = word_tokenize(u) # compare the results with the manual splitter
    word_tokens = u.split()
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

    fp = open('new_docs/docs_new/'+doc,'r')
    tmp_doc = fp.read().split("Body:")
    title_p = []
    body_p = []
    title_p.append(tmp_doc[0].replace("Title:", "").replace("\n", ""))
    try:
        body_p.append(tmp_doc[1].replace("\n", ""))
    except:
        body_p = [] # no body

    title = []
    body = []


    for x  in title_p:
        u = strip_special_chars(x)
        if re.match(r'.*[a-z]+.*', u):
            title.append(u)

    for x  in body_p:
        u = strip_special_chars(x)
        if re.match(r'.*[a-z]+.*', u): # filter empty lines
            body.append(u)
            
    return title,body

    
# raw_docs can be titles or bodies
def doc_process_tf_idf(raw_docs):
#    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_docs]
    gen_docs = [w.split() for w in raw_docs]
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


def write_body_features1(body_hist,bstring,key):
    global train_body_f1
    global test_body_f1
    ctr = 1
    for count in body_hist:
        #body_features.write(str(ctr)+':'+str(count)+' ')
        bstring = bstring + str(ctr)+':'+str(count)+' '
        ctr = ctr+1
    #body_features.write('\n')
    if key - 300 <= 113:
        train_body_f1.append(bstring)
    else:
        test_body_f1.append(bstring)
    
def thread1():
    global train_title_f1
    global test_title_f1
    global qrels_map
    global title_set
    global body_set
    global queries

    print('----------- '+str(len(qrels_map))+' '+str(len(title_set))+' '+str(len(body_set))+' '+str(len(queries)))
    
    keys = [str(x) for x in range(301,339)]
    for key in keys:
        for doc in qrels_map[key]:
            docname = doc[0]
            relevancy = doc[1]
            #if os.path.isfile('test-data/example/'+docname) == False:
            if os.path.isfile('new_docs/docs_new/'+docname) == False:
                continue

            title = title_set[docname]
            body = body_set[docname]

            query = queries[key]
            title_hist = get_hist(query,title)
            body_hist = get_hist(query,body)
            #print(title_hist)

            #title_features.write(relevancy+' qid:'+key+' ')
            tstring = relevancy+' qid:'+key+' '

            ctr = 1
            for count in title_hist:
                #title_features.write(str(ctr)+':'+str(count)+' ')
                tstring = tstring + str(ctr)+':'+str(count)+' '
                ctr = ctr+1
            #title_features.write('\n')
            if int(key) -300 <= 113:
                train_title_f1.append(tstring)
            else:
                test_title_f1.append(tstring)

            #body_features.write(relevancy+' qid:'+key+' ')
            bstring = relevancy+' qid:'+key+' '
            write_body_features1(body_hist,relevancy+' qid:'+key+' ',int(key))
            #print('body '+str(fctr))


            if int(key) - 300 <= 113 and relevancy=='1':
                print('here')
                sys.exit()
                #generate augmented features for body
                body_hist1 = get_hist(query,body,0.7)
                #print('body1 '+str(fctr))
                body_hist2 = get_hist(query,body,0.7)
                #print('body2 '+str(fctr))
                body_hist3 = get_hist(query,body,0.7)
                #print('body3 '+str(fctr))
                body_hist4 = get_hist(query,body,0.7)
                #print('body4 '+str(fctr))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features1(body_hist1,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features1(body_hist2,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features1(body_hist3,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features1(body_hist4,relevancy+' qid:'+key+' ',int(key))

    tr1 = open('title_train1','wb')
    te1 = open('title_test1','wb')
    br1 = open('body_train1','wb')
    be1 = open('body_test1','wb')

    pickle.dump(train_title_f1,tr1)
    tr1.close()
    pickle.dump(test_title_f1,te1)
    te1.close()
    pickle.dump(train_body_f1,br1)
    br1.close()
    pickle.dump(test_body_f1,be1)
    be1.close()
                

def write_body_features2(body_hist,bstring,key):
    global train_body_f2
    global test_body_f2
    ctr = 1
    for count in body_hist:
        #body_features.write(str(ctr)+':'+str(count)+' ')
        bstring = bstring + str(ctr)+':'+str(count)+' '
        ctr = ctr+1
    #body_features.write('\n')
    if key - 300 <= 113:
        train_body_f2.append(bstring)
    else:
        test_body_f2.append(bstring)
    
def thread2():
    global train_title_f2
    global test_title_f2
    global qrels_map
    global title_set
    global body_set
    global queries

    print('----------- '+str(len(title_set)))

    keys = [str(x) for x in range(339,377)]
    for key in keys:
        for doc in qrels_map[key]:
            docname = doc[0]
            relevancy = doc[1]
            #if os.path.isfile('test-data/example/'+docname) == False:
            if os.path.isfile('new_docs/docs_new/'+docname) == False:
                continue

            title = title_set[docname]
            body = body_set[docname]

            query = queries[key]
            title_hist = get_hist(query,title)
            body_hist = get_hist(query,body)
            #print(title_hist)

            #title_features.write(relevancy+' qid:'+key+' ')
            tstring = relevancy+' qid:'+key+' '

            ctr = 1
            for count in title_hist:
                #title_features.write(str(ctr)+':'+str(count)+' ')
                tstring = tstring + str(ctr)+':'+str(count)+' '
                ctr = ctr+1
            #title_features.write('\n')
            if int(key) -300 <= 113:
                train_title_f2.append(tstring)
            else:
                test_title_f2.append(tstring)

            #body_features.write(relevancy+' qid:'+key+' ')
            bstring = relevancy+' qid:'+key+' '
            write_body_features2(body_hist,relevancy+' qid:'+key+' ',int(key))
            #print('body '+str(fctr))


            if int(key) - 300 <= 113 and relevancy=='1':
                #generate augmented features for body
                body_hist1 = get_hist(query,body,0.7)
                #print('body1 '+str(fctr))
                body_hist2 = get_hist(query,body,0.7)
                #print('body2 '+str(fctr))
                body_hist3 = get_hist(query,body,0.7)
                #print('body3 '+str(fctr))
                body_hist4 = get_hist(query,body,0.7)
                #print('body4 '+str(fctr))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features2(body_hist1,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features2(body_hist2,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features2(body_hist3,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features2(body_hist4,relevancy+' qid:'+key+' ',int(key))
    tr2 = open('title_train2','wb')
    te2 = open('title_test2','wb')
    br2 = open('body_train2','wb')
    be2 = open('body_test2','wb')

    pickle.dump(train_title_f2,tr2)
    tr2.close()
    pickle.dump(test_title_f2,te2)
    te2.close()
    pickle.dump(train_body_f2,br2)
    br2.close()
    pickle.dump(test_body_f2,be2)
    be2.close()

def write_body_features3(body_hist,bstring,key):
    global train_body_f3
    global test_body_f3
    ctr = 1
    for count in body_hist:
        #body_features.write(str(ctr)+':'+str(count)+' ')
        bstring = bstring + str(ctr)+':'+str(count)+' '
        ctr = ctr+1
    #body_features.write('\n')
    if key - 300 <= 113:
        train_body_f3.append(bstring)
    else:
        test_body_f3.append(bstring)
    
def thread3():
    global train_title_f3
    global test_title_f3
    global qrels_map
    global title_set
    global body_set
    global queries

    print('----------- '+str(len(title_set)))

    keys = [str(x) for x in range(377,415)]
    for key in keys:
        for doc in qrels_map[key]:
            docname = doc[0]
            relevancy = doc[1]
            #if os.path.isfile('test-data/example/'+docname) == False:
            if os.path.isfile('new_docs/docs_new/'+docname) == False:
                continue

            title = title_set[docname]
            body = body_set[docname]

            query = queries[key]
            title_hist = get_hist(query,title)
            body_hist = get_hist(query,body)
            #print(title_hist)

            #title_features.write(relevancy+' qid:'+key+' ')
            tstring = relevancy+' qid:'+key+' '

            ctr = 1
            for count in title_hist:
                #title_features.write(str(ctr)+':'+str(count)+' ')
                tstring = tstring + str(ctr)+':'+str(count)+' '
                ctr = ctr+1
            #title_features.write('\n')
            if int(key) -300 <= 113:
                train_title_f3.append(tstring)
            else:
                test_title_f3.append(tstring)

            #body_features.write(relevancy+' qid:'+key+' ')
            bstring = relevancy+' qid:'+key+' '
            write_body_features3(body_hist,relevancy+' qid:'+key+' ',int(key))
            #print('body '+str(fctr))


            if int(key) - 300 <= 113 and relevancy=='1':
                #generate augmented features for body
                body_hist1 = get_hist(query,body,0.7)
                #print('body1 '+str(fctr))
                body_hist2 = get_hist(query,body,0.7)
                #print('body2 '+str(fctr))
                body_hist3 = get_hist(query,body,0.7)
                #print('body3 '+str(fctr))
                body_hist4 = get_hist(query,body,0.7)
                #print('body4 '+str(fctr))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features3(body_hist1,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features3(body_hist2,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features3(body_hist3,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features3(body_hist4,relevancy+' qid:'+key+' ',int(key))
    tr3 = open('title_train3','wb')
    te3 = open('title_test3','wb')
    br3 = open('body_train3','wb')
    be3 = open('body_test3','wb')

    pickle.dump(train_title_f3,tr3)
    tr3.close()
    pickle.dump(test_title_f3,te3)
    te3.close()
    pickle.dump(train_body_f3,br3)
    br3.close()
    pickle.dump(test_body_f3,be3)
    be3.close()
    


def write_body_features4(body_hist,bstring,key):
    global train_body_f4
    global test_body_f4
    ctr = 1
    for count in body_hist:
        #body_features.write(str(ctr)+':'+str(count)+' ')
        bstring = bstring + str(ctr)+':'+str(count)+' '
        ctr = ctr+1
    #body_features.write('\n')
    if key - 300 <= 113:
        train_body_f4.append(bstring)
    else:
        test_body_f4.append(bstring)
    
def thread4():
    global train_title_f4
    global test_title_f4
    global qrels_map
    global title_set
    global body_set
    global queries

    print('----------- '+str(len(title_set)))

    keys = [str(x) for x in range(415,451)]
    for key in keys:
        for doc in qrels_map[key]:
            docname = doc[0]
            relevancy = doc[1]
            #if os.path.isfile('test-data/example/'+docname) == False:
            if os.path.isfile('new_docs/docs_new/'+docname) == False:
                continue

            title = title_set[docname]
            body = body_set[docname]

            query = queries[key]
            title_hist = get_hist(query,title)
            body_hist = get_hist(query,body)
            #print(title_hist)

            #title_features.write(relevancy+' qid:'+key+' ')
            tstring = relevancy+' qid:'+key+' '

            ctr = 1
            for count in title_hist:
                #title_features.write(str(ctr)+':'+str(count)+' ')
                tstring = tstring + str(ctr)+':'+str(count)+' '
                ctr = ctr+1
            #title_features.write('\n')
            if int(key) -300 <= 113:
                train_title_f4.append(tstring)
            else:
                test_title_f4.append(tstring)

            #body_features.write(relevancy+' qid:'+key+' ')
            bstring = relevancy+' qid:'+key+' '
            write_body_features4(body_hist,relevancy+' qid:'+key+' ',int(key))
            #print('body '+str(fctr))


            if int(key) - 300 <= 113 and relevancy=='1':
                #generate augmented features for body
                body_hist1 = get_hist(query,body,0.7)
                #print('body1 '+str(fctr))
                body_hist2 = get_hist(query,body,0.7)
                #print('body2 '+str(fctr))
                body_hist3 = get_hist(query,body,0.7)
                #print('body3 '+str(fctr))
                body_hist4 = get_hist(query,body,0.7)
                #print('body4 '+str(fctr))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features4(body_hist1,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features4(body_hist2,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features4(body_hist3,relevancy+' qid:'+key+' ',int(key))

                body_features.write(relevancy+' qid:'+key+' ')
                write_body_features4(body_hist4,relevancy+' qid:'+key+' ',int(key))
    tr4 = open('title_train4','wb')
    te4 = open('title_test4','wb')
    br4 = open('body_train4','wb')
    be4 = open('body_test4','wb')

    pickle.dump(train_title_f4,tr4)
    tr4.close()
    pickle.dump(test_title_f4,te4)
    te4.close()
    pickle.dump(train_body_f4,br4)
    br4.close()
    pickle.dump(test_body_f4,be4)
    be4.close()
    
queries = parse_queries()
qrels_map = get_qrels_map()

title_features = open('test-data/features/qtitle.txt','w')
body_features = open('test-data/features/qbody.txt','w')

examples=0
title_set = {}
body_set = {}

'''dir = sort(os.listdir('test-data/docs')
#dir = os.listdir('test-data/example')
fctr = 0
for fname in dir:
    fctr = fctr + 1
    p = round(fctr*100/337719.0, 2)
    #print('processing '+str(fctr)+'/337719 -- '+str(p)+'%', end='\r')
    print('processing '+str(fctr))
    title,body = parse_doc(fname)
    title_set[fname] = title
    body_set[fname] = body

#build tf-idf values for words in title and body
arg_t = []
arg_b = []
idx = 0
index_map_t = {}
index_map_b = {}'''

'''print('start processing docs')
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
print('end processing docs')'''

'''print('corpus creation begins')        
corpus_t, dictionary_t, tf_idf_t = doc_process_tf_idf(arg_t)
corpus_b, dictionary_b, tf_idf_b = doc_process_tf_idf(arg_b)
print('corpus creation ends')        

arg_t.clear()
arg_b.clear()

#qid: tfidf values for each dcument wrt to the query as numpy array
query_tf_idf_title = {}
query_tf_idf_body = {}

print('tf-idf calculation begins')
for qid in queries:
    # need to build the query tf-idf matrix q:doc1,doc2,...
    query_tf_idf_title[qid] = single_query_docs_tf_idf(queries[qid],corpus_t,dictionary_t,tf_idf_t)*10
    query_tf_idf_body[qid] = single_query_docs_tf_idf(queries[qid],corpus_b,dictionary_b,tf_idf_b)*10
print('tf-idf calculation ends')'''

print('begin augmentation')

ts = open('title_set','wb')
pickle.dump(title_set,ts)
ts.close()

bs = open('body_set','wb')
pickle.dump(body_set,bs)
bs.close()


ts = open('title_set_new','rb')
bs = open('body_set_new','rb')

title_set = pickle.load(ts)
body_set = pickle.load(bs)

ts.close()
bs.close()

print(len(title_set))
print(title_set['FT911-1237'])


model = gensim.models.KeyedVectors.load_word2vec_format('/home/dheeraj/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
#model.similarity(w1='clean',w2='neat')

train_title_f1=[]
train_body_f1 = []

test_title_f1=[]
test_body_f1 = []

train_title_f2=[]
train_body_f2 = []

test_title_f2=[]
test_body_f2 = []

train_title_f3=[]
train_body_f3 = []

test_title_f3=[]
test_body_f3 = []

train_title_f4=[]
train_body_f4 = []

test_title_f4=[]
test_body_f4 = []


for keyw in qrels_map:
    qrels_map[keyw] = sorted(qrels_map[keyw],key=lambda x:x[0])

        
                

'''ts = open('title_set','wb')
pickle.dump(title_set,ts)
ts.close()'''

Thread(target = thread1).start()
Thread(target = thread2).start()
Thread(target = thread3).start()
Thread(target = thread4).start()

#title_set.clear()
#body_set.clear()


'''print('begin writing titles')
title_features.write('\n'.join(title_f))
title_features.close()
print('end writing titles')

print('begin writing body')
body_features.write('\n'.join(body_f))
body_features.close()
print('end writing body')

#IMPORTANT: remember to do a shuffling during training/ during generation?
print('produced features for '+str(examples)+' training points')'''
