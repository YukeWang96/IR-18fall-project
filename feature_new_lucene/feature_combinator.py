import pickle

with open("qd_dict.bin", "rb") as pf:
    qd_dict = pickle.load(pf)

#----------------for output title result
# tf_idf_file = "00_tf-idf_title.out"
# bm25_file = "00_bm25_title.out"

# f_output_title = "02_title_features.log"

# fp_tf_idf_title = open(tf_idf_file, "r")
# fp_bm25_title = open(bm25_file, "r")
# f_output = open(f_output_title, "w")

#-----------------for output body result
tf_idf_file = "01_tf-idf_body.out"
bm25_file = "01_bm25_body.out"
min_dist_file = "min_dist.log"

f_output_body = "02_body_features.log"

fp_tf_idf_title = open(tf_idf_file, "r")
fp_bm25_title = open(bm25_file, "r")
fp_min_distance  = open(min_dist_file, "r")
f_output = open(f_output_body, "w")
#---------------------------------------------

qrd_tf_idf_title = []
for qry_no in qd_dict.keys():
    tmp_qd_list = qd_dict[qry_no]
    tmp_qd_dict = {}
    tmp_score_qd = []

    for qry_no_doc in tmp_qd_list:
        tmp_qd_dict[qry_no_doc] = '0'

    fp_tf_idf_title.seek(0)
    for line in fp_tf_idf_title:
        line_list = line.strip("\n").split(" ")
        if line_list[1] == "None":
            continue

        if str(qry_no) == str(line_list[0]):
            if line_list[2] in tmp_qd_list:
                tmp_qd_dict[line_list[2]] = str(line_list[4])
    
    tmp_result_tf_idf = [tmp_qd_dict[key] for key in sorted(tmp_qd_list)]
    qrd_tf_idf_title.append(tmp_result_tf_idf)
    print("["+ str(qry_no) + "]")

fp_tf_idf_title.close()

qrd_bm25_title = []
for qry_no in qd_dict.keys():
    tmp_qd_list = qd_dict[qry_no]
    tmp_qd_dict = {}

    for qry_no_doc in qd_dict[qry_no]:
        tmp_qd_dict[qry_no_doc] = '0'

    fp_bm25_title.seek(0)
    for line in fp_bm25_title:
        line_list = line.strip("\n").split(" ")
        if line_list[1] == "None":
            continue
        if str(qry_no) == line_list[0]:
            if line_list[2] in tmp_qd_list:
                tmp_qd_dict[line_list[2]] = str(line_list[4])
        
    tmp_result = [tmp_qd_dict[key] for key in sorted(tmp_qd_list)]
    qrd_bm25_title.append(tmp_result)
    print("["+ str(qry_no) + "]")

fp_bm25_title.close()

# output title result of tf-idf and bm25
# qid = 301
# for tf_idf_list, bm25_list in zip(qrd_tf_idf_title, qrd_bm25_title):
#     for tf_idf, bm25 in zip(tf_idf_list, bm25_list):
#         result = str(qid) + " "
#         result += "tf-idf:" + tf_idf + " "
#         result += "bm25:" + bm25 + "\n"
#         f_output.write(result)
#     qid += 1

# f_output.close()

# # output body resulf of tf-idf, bm25 and min_distance
qid = 301
for tf_idf_list, bm25_list, minD in zip(qrd_tf_idf_title, qrd_bm25_title, fp_min_distance):
    for tf_idf, bm25, min_di in zip(tf_idf_list, bm25_list, minD.split(" ")):
        result = str(qid) + " "
        result += "tf-idf:" + tf_idf + " "
        result += "bm25:" + bm25 + " "
        result += "min_dist:" + min_di + "\n"
        f_output.write(result)
    qid += 1

f_output.close()