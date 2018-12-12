# IR-18fall-project

This is the code repository for `trec-disk4-5` data processing, feature generation of `tf-idf`, `bm25` and `minimum distance proximity` for both article titles and bodies.

# Data file generating
+ Using `parsing_xml.py` to generate the seperate files for each articles in `trec-disk-4-5.xml`, use the `DOCNO` as the files' name, and inside each file, it contain the title and body part of each article.
+ For those articles which only have title part, compensate the body part by using the title part.
+ Store all those generated files in step 1 into the same directory called `data`.

# feature_old_python--using Python to generate similarity features
+ Read in all the files under the directory `data`.
+ Use `Python` natural language processing package--`gensim` to generate the tf-idf and bm25 similarity.
+ `qd-tf-idf.py`: get the tf-idf similarity between a query and all documents.
+ `qd-bm25.py`: get the bm25 similarity between a query and all documents.
+ `qd_min_distance.py`: get the minimum distance proximity between a query and all documents.

# feature_new_lucene--using Lucene to generate similarity features
+ Read in all the files under the directory `data`.
+ Use `Lucene-7.5.0` and `tec-demo` as the tool for getting the tf-idf and bm25 values between a query and all doucments.
