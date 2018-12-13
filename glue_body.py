# script that combines the histogram and tf-idf, bm25 features
#author: dheeraj baby

bm = open('02_body_features.log','r')
hist1 = open('feature_body_train','r')
hist2 = open('feature_body_test','r')


lines_bm = bm.readlines()
train_hist = hist1.readlines()
test_hist = hist2.readlines()

bm.close()
hist1.close()
hist2.close()

lines_bm = [x.strip()+' ' for x in lines_bm]
train_hist = [x.strip()+' ' for x in train_hist]
test_hist = [x.strip()+' ' for x in test_hist]

new_train = []
new_test = []

i = 0
j = 0

while True:
    try:
        line = train_hist[i]
    except:
        try:
            line = test_hist[i-len(train_hist)]
        except:
            break
        
    tmp = line[:20].split(':')
    relevancy = tmp[0].split()[0]
    qid = int(tmp[1].split()[0])

    line = lines_bm[j]
    tmp = line.split(':')
    tfidf = tmp[1].split()[0]
    bm25 = tmp[2].split()[0]
    qid2 = int(tmp[0].split()[0])

    j = j+1
    
    assert qid2 == qid
    
    if qid <= 413 and relevancy == '1':
        for k in range(5):
            hline = train_hist[i]
            l = hline+'51:'+tfidf+' 52:'+bm25
            new_train.append(l)
            i = i+1
    elif qid <= 413 and relevancy == '0':
        hline = train_hist[i]
        l = hline+'51:'+tfidf+' 52:'+bm25
        new_train.append(l)
        i = i+1
    else:
        # test examples
        hline = test_hist[i-len(train_hist)]
        l = hline+'51:'+tfidf+' 52:'+bm25
        new_test.append(l)
        i = i+1
        
        
    
    
print(str(len(new_train))+' '+str(len(train_hist)))
print(str(len(new_test))+' '+str(len(test_hist)))

train = open('full_train_body','w')
train.write('\n'.join(new_train))
train.close()

test = open('full_test_body','w')
test.write('\n'.join(new_test))
test.close()

