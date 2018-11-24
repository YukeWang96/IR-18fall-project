# coding = "UTF-8"
import xml.etree.ElementTree as ET

# with open(u"trec-disk4-5_processed.xml", 'r') as ip:
#     input_xml = ip.read()

# stuff = ET.fromstring(input_xml)
tree = ET.parse("trec-disk4-5_processed.xml")
root = tree.getroot()
lst = root.findall('DOC')
print('User count:', len(lst))

counter = 0

for item in lst:
    # output_str = 'DocNo: ' + item.find('DOCNO').text + '\n'
    output_str = ""

    title = item.find('HEADLINE')
    if title is None or len(title.text) == 0:
        continue
    
    docnum = item.find('DOCNO').text.strip(" ").strip("'")
    fp = open("docs/" + docnum, 'w')

    if len(title.text) < 2:
        title_text_recur = item.findall('HEADLINE/P')
        result = ""
        for txt in title_text_recur:
            result += txt.text
        # print('Title: ', result)
        # fp.write('Title: ' + result)
        output_str += 'Title: ' + result.lower() + '\n'
    else:
        # print('Title: ', title.text)
        # fp.write('Title: ' + title.text)
        output_str += 'Title: ' + (title.text).lower() + '\n'


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
