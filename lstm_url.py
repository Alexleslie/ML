import tensorflow as tf 
from tensorflow.contrib import rnn
from deal_sample import deal

bad_file_name = "badqueries.txt"
good_file_name = "goodqueries.txt"

sample_num = 10000


bad = open(bad_file_name,encoding='utf-8').readlines()[:45000]   # 前四百个样本
good = open(good_file_name).readlines()[:45000]
bad = deal(bad)[:40000]  # 去掉换行符
good = deal(good)[:40000]
print(len(bad), len(good))