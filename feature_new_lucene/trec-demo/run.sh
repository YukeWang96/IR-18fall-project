rm -rf index
java -cp "bin:lib/*" IndexTREC -docs output_files_title.xml
java -cp "bin:lib/*" BatchSearch -index index -queries test-data/title-queries.301-450 -simfn default > 00_tf-idf_title.out
java -cp "bin:lib/*" BatchSearch -index index -queries test-data/title-queries.301-450 -simfn bm25 > 00_bm25_title.out

rm -rf index
java -cp "bin:lib/*" IndexTREC -docs output_files_body.xml
java -cp "bin:lib/*" BatchSearch -index index -queries test-data/title-queries.301-450 -simfn default > 01_tf-idf_body.out
java -cp "bin:lib/*" BatchSearch -index index -queries test-data/title-queries.301-450 -simfn bm25 > 01_bm25_body.out