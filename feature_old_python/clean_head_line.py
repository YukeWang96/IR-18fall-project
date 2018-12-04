fp = open("trec-disk4-5_processed.xml", 'r')
fo = open("trec-disk4-5_processed_out.xml", 'w')

counter = 0

for line in fp:
    l = line.replace("<AU>", "")
    l = l.replace("</AU>", "") 
    l = l.replace("<DATE1>", "")
    l = l.replace("</DATE1>", "")  
    fo.write(l) 
    counter += 1
    print(counter)

fp.close()
fo.close()