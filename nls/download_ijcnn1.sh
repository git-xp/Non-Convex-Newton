#!/usr/bin/env bash

# Linux User
# wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2 
# wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2

# Mac User
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2" -o "ijcnn1.bz2" 
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2" -o "ijcnn1.t.bz2"

bzip2 -d ijcnn1.bz2
bzip2 -d ijcnn1.t.bz2

rm ijcnn1.bz2 ijcnn1.t.bz2