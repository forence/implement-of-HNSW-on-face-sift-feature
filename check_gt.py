import struct
import numpy as np
import os

# read gt.bin
num = 1000
count = 1
top = 20
root_path = "/home/zhangziqi/storage/project/hnsw/small_face/face_data"
with open(os.path.join(root_path, 'face_sift_gt_.bin'), 'rb') as db_file:
    line = db_file.read((1 + num) * 4)
    format_ = 'i' * (num + 1)
    while count < 2:
        tmp = struct.unpack(format_, line)
        # print("query ", count, " : ", tmp[1:])
        line = db_file.read((1 + num) * 4)
        count += 1

# from index.txt find index
with open(os.path.join(root_path, 'face_sift_query_index.txt'), 'r') as query_idx_file:
    query_idx = query_idx_file.readlines()
with open(os.path.join(root_path, 'face_sift_db_index.txt'), 'r') as db_idx_file:
    db_idx = db_idx_file.readlines()
# query_idx = np.loadtxt('./face_sift_query_index.txt', dtype={'names': ('num', 'index'), 'formats': ('i4', 'S30')})
# db_idx = np.loadtxt('./face_sift_db_index.txt', dtype={'names': ('num', 'index'), 'formats': ('i4', 'S30')})
print "query ID : ", query_idx[0],
for i in range(top, top + 10):
    print "db top ", i, " ID : ", db_idx[tmp[i]],   
