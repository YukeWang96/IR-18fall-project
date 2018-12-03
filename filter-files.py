import os
import pickle
from shutil import copy2

data_file = "docs_new"

dir = os.listdir(data_file)

with open("qd_dict.bin", "rb") as fqd:
    qd_dict = pickle.load(fqd)

files_set=[]
for key in qd_dict.keys():
    files_set += qd_dict[key]

files_keep_set = set(files_set)
total_size = len(files_keep_set)

os.mkdir('docs_new_small')
print("create doc_new_small....")

counter = 0
for fp in dir:
    if fp in files_keep_set:
        copy2(data_file + "/" + fp, 'docs_new_small')
        counter += 1
        print(str(counter) + "/" + str(total_size))

