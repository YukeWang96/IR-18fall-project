import pickle

counter = 0
with open("qd_dict.bin", "rb") as fp:
    qd_dict = pickle.load(fp)

for i in qd_dict.keys():
    counter += len(qd_dict[i])
    # print(qd_dict[i])

print(counter)