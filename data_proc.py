# coding="UTF-8"

fp = open(u"lines-trec45.txt", "r")
counter = 0

for line in fp:
    temp = line.split("\t")
    docno = temp[0]
    if docno.startswith("FT"):
        title = "Title: " + temp[1][12:]
    else:
        title = "Title: " + temp[1]
    body = "Body: " + temp[2]
    fo = open("docs//" + temp[0], "w")
    # print("docno: " + docno + "\n")
    # print(title + "\n\n" + body)
    fo.write(title + "\n\n" + body)
    counter += 1
    print(counter)
    # if counter == 100:
    #     break
