import struct
import numpy as np
import time
import os
"""
brute-force method for computing ground truth
because each feature has been normalized, so there is a trick to direct compute two feature vectors' dot
the bigger value it is the distance is smaller
note: In our face-dataset each feature vector contains 
        1 identifier : 1*4 bytes(int) and 128 dimensions vector :128*4 bytes(float)
"""

root_path = "/home/zhangziqi/storage/project/hnsw/small_face/face_data"
query_path = os.path.join(root_path, "face_sift_query.bin")
db_path = os.path.join(root_path, "face_sift_db.bin")

vecdim = 128
query_num = os.path.getsize(query_path) // ((vecdim + 1) * 4)
db_num = os.path.getsize(db_path) // ((vecdim + 1) * 4)

# read query.bin and store feature vector in query_arr
print("Reading query ...")
query_arr = np.zeros((query_num, vecdim), dtype=np.float32)
query_idx = 0
with open(query_path, 'rb') as query_file:
    query = query_file.read((vecdim + 1) * 4)
    format_ = 'i' + 'f' * vecdim
    while query:
        tmp = struct.unpack(format_, query)
        dim = tmp[0]
        query_vec = np.array(tmp[1:], dtype=np.float32)
        if dim != vecdim:
            print('file error')
            exit(1)
        else:
            query_arr[query_idx] = query_vec
        query = query_file.read((vecdim + 1) * 4)
        if query_idx % 5000 == 0:
            print "\r%d" % query_idx,
        query_idx += 1

print "\nquery number = %d" % query_num


# read database.bin and store feature vector in db_arr
print("Reading db ...")
db_arr = np.zeros((db_num, vecdim), dtype=np.float32)
db_idx = 0
with open(db_path, 'rb') as db_file:
    db = db_file.read((vecdim + 1) * 4)
    format_ = 'i' + 'f' * vecdim
    while db:
        tmp = struct.unpack(format_, db)
        dim = tmp[0]
        db_vec = np.array(tmp[1:], dtype=np.float32)
        if dim != vecdim:
            print('file error')
            exit(2)
        else:
            db_arr[db_idx] = db_vec
        db = db_file.read((vecdim + 1) * 4)
        if db_idx % 100 == 0:
            print "\r%d" % db_idx,
        db_idx += 1

    print "\ndatabase number = %d" % db_num


# compute distance
time_start = time.time()
top = 1000
gt = np.zeros(top, dtype=np.int32)
query2db = np.zeros(db_num, dtype=np.float32)
gt_bin_file = open(os.path.join(root_path, "face_sift_gt.bin"), 'wb')

for query_idx, query in enumerate(query_arr):
    print ">>dot"
    query2db = np.dot(db_arr, query)
    print ">>argsort"
    gt = np.argpartition(query2db, [x for x in range(-1, -(top + 1), -1)])[-1:-(top + 1):-1]
   # gt = np.argsort(query2db)[:-(top + 1):-1]
    format_ = 'i' * (top+1)
    print ">>pack"
    line_binary = struct.pack(format_, top, *gt)
    gt_bin_file.write(line_binary)
    print "\r%d" % query_idx,
gt_bin_file.close()
time_end = time.time()
print "Total time = %f"%(time_end - time_start)

