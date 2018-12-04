
fp=open("output_files_title.xml", "r")

counter = 0
for line in fp:
    if "<DOC>" in line:
        counter += 1

print(counter)