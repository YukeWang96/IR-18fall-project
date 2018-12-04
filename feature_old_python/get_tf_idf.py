
# fp=open("BM25_result_title.log")
fp=open("min_dist.log")

count = 0

for i in fp:
    str_list = list(filter(None, i.strip("\n").split(",")))
    print(str_list)
    count += len(str_list)
    # break

print("total_num: " + str(count))
fp.close()