
generate_title = True
generate_body = True

counter = 0
if generate_title:
    fp_tf_idf_title = open("TF-IDF_result_title.log", "r")
    fp_bm25_title = open("BM25_result_title.log", "r")

    f_output = open("00_title_features.log", "w")
    qid = 301

    for line_tf_idf, line_bm25 in zip(fp_tf_idf_title, fp_bm25_title):
        # if len(line_tf_idf) == 0 or len(line_bm25) == 0:
        #     break
        tf_idf_score = list(filter(None, line_tf_idf.strip("\n").split(" ")))
        bm25_score = list(filter(None, line_bm25.strip("\n").split(" ")))

        # print(str(len(tf_idf_score)) + "---" + str(len(bm25_score)))
        for idx in range(len(tf_idf_score)):
            # if len(tf_idf_score[idx]) == 0:
            #     break
            result = ""
            result += "qid:" + str(qid) + " " + "tf-idf_title:" + str(tf_idf_score[idx]) + " " + "bm25_title:" + str(bm25_score[idx]) + "\n"
            f_output.write(result)
            # counter += 1

        qid += 1
    # print(counter)
    
    fp_tf_idf_title.close()
    fp_bm25_title.close()
    f_output.close()


if generate_body:
    fp_tf_idf_body = open("TF-IDF_result_body.log", "r")
    fp_bm25_body = open("BM25_result_body.log", "r")
    min_dist_body = open("min_dist.log", "r")

    f_output_body = open("00_body_features.log", "w")
    qid = 301

    for line_tf_idf, line_bm25, line_min_dist in zip(fp_tf_idf_body, fp_bm25_body, min_dist_body):
        # if len(line_tf_idf) == 0 or len(line_bm25) == 0 or len(line_min_dist) == 0:
        #     break

        tf_idf_score = list(filter(None, line_tf_idf.strip("\n").split(" ")))
        bm25_score = list(filter(None, line_bm25.strip("\n").split(" ")))
        min_dist = list(filter(None, line_min_dist.strip("\n").split(" ")))

        for idx in range(len(tf_idf_score)):
            # if len(tf_idf_score[idx]) == 0:
            #     break
            result = "qid:" + str(qid) + " "
            result += "tf-idf_body:" + str(tf_idf_score[idx]) + " "
            result += "bm25_body:" + str(bm25_score[idx]) + " "
            result += "min_dist:" + str(min_dist[idx]) + "\n"
            f_output_body.write(result)
            
        qid += 1

    fp_tf_idf_body.close()
    fp_bm25_body.close()
    min_dist_body.close()
    f_output_body.close()