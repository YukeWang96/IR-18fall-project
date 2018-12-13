#!/bin/bash

# bash script to do parameter tuning for MART model'
# author: dheeraj baby

T=0
L=0
LR=0

max=0

t=(20 100 200 300 400 500 600 700 800 900 1000)
l=(10 15 20 25 30 35 40 45 50 100)
lr=(0.05 0.1 0.2 0.4 0.6 0.8 1)

#t=(20)
#l=(10)
#lr=(0.1)


for i in ${t[@]}; do
    for j in ${l[@]}; do
	for k in ${lr[@]}; do
	    java -jar RankLib-2.10.jar -ranker 0 -train full_train_body -test full_test_body -metric2t NDCG@10 -metric2T NDCG@10 -kcv 5 -tree ${i} -leaf ${j}   -shrinkage ${k} -estop 100  -silent > templog0
	    

	    num=$(tail templog0  | grep 'Total' | awk '{print $4}')
	    res=$(echo ${num}'>'${max} | bc -l)
	    if [ "$res" -gt 0 ]
	    then
		max=$num
		T=$i
		L=$j
		LR=$k
		echo "max value is $max attained at $T $L $LR"
	    fi
	done
    done
done

echo "** max value is $max attained at $T $L $LR"
