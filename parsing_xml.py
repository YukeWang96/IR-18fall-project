# coding = "UTF-8"
import xml.etree.ElementTree as ET
import pickle

# with open("xml_tree.bin", "rb") as finput:
#     tree = pickle.load(finput)
    
tree = ET.parse("trec-disk4-5_processed_out.xml")
root = tree.getroot()
lst = root.findall('DOC')

#with open("xml_tree.bin", "wb") as foutput:
#    pickle.dump(tree, foutput)

# print('User count:', len(lst))
counter = 0

for item in lst:
    # output_str = 'DocNo: ' + item.find('DOCNO').text + '\n'
    output_str = ""

    # if (title_headline is None and title_header is None) or len(title.text) == 0:
    #     continue
    
    docnum = item.find('DOCNO').text.strip(" ").strip("'")

    # if docnum.startswith("CR"):
    #     continue

    fp = open("docs_new/" + docnum, 'w')

    title_headline = item.find('HEADLINE')
    title_header = item.find('HEADER')

    if title_headline is not None:
        if len(title_headline.text) < 2:
            title_text_recur = item.findall('HEADLINE/P')
            result = ""
            for txt in title_text_recur:
                result += txt.text
        else:
            result = title_headline.text
            # print('Title: ', result)
            # fp.write('Title: ' + result)
        output_str += 'Title: ' + result.lower() + '\n'
    
    if title_header is not None:
        output_str += 'Title: ' + title_header.text.lower() + '\n'


    body_text = item.find('TEXT')
    if body_text is not None:
        if len(body_text.text) < 3:
            body_text_recur = item.findall('TEXT/P')
            result = ""
            for txt in body_text_recur:
                result += txt.text
                # print('Body: ', result)
                # fp.write('Body: ' + result)
            output_str += 'Body: ' + result.lower() + '\n'
        else:
            output_str += 'Body: ' + (body_text.text).lower() + '\n'
            # print('Body: ', body_text.text)
            # fp.write('Body: ' + body_text.text + '\n')

    fp.write(output_str)
    counter += 1
    print(docnum)
    print(counter)
