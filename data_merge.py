import struct
import numpy as np
import glob
"author:zzq"

files_path = glob.glob('/raid/yuanyong/project/face_ann_search/face_data/*.*')

vecdim = 128

# read all files
file_index = 1
print('Reading all files ...')
all_dect = []
for path in files_path:
    with open(path, 'r') as file_:
        content = file_.readlines()
        all_dect += content
        print "\r------%d / %d" % (file_index, len(files_path)),
        file_index += 1
total_vec = len(all_dect)
total_file = len(files_path)
print "\ntotal vectors : %d" % total_vec
print "total files : %d" % total_file

# shuffle
print("Shuffling ...")
rand_idx = np.arange(total_vec)
np.random.seed(1024)
np.random.shuffle(rand_idx)


# read each lines in order to obtain face_ID and face_vec furthermore write them in .txt and .bin file
db_bin_file = open('face_sift_db.bin', 'wb')
db_index_file = open('face_sift_db_index.txt', 'w')
query_bin_file = open('face_sift_query.bin', 'wb')
query_index_file = open('face_sift_query_index.txt', 'w')

# build query and its index
query_index = 1
query_num = 10000
print('Writing query ...')
for i in range(query_num):
    line = all_dect[rand_idx[i]]
    face_ID, vec_str = line.strip().split(':')
    vec_str = vec_str[:-1].split(',')
    if len(vec_str) != vecdim:
        print('file_error')
        continue
    else:
        query_index_file.write("%s %s\n" % (str(query_index), face_ID))
        query_index += 1
        face_vec = [float(x) for x in vec_str]
        format_ = 'i' + 'f' * vecdim
        line_binary = struct.pack(format_, vecdim, *face_vec)  # len(bytes) = (1+128)*4=516
        query_bin_file.write(line_binary)
    if query_index % 1000 == 0:
	print "\r%d / %d" % (query_index, query_num),
query_bin_file.close()
query_index_file.close()


# build database and its index
print('\nWriting database ...')
db_index = 1
db_num = total_vec - query_num

for j in range(query_num, total_vec):
    line = all_dect[rand_idx[j]]
    face_ID, vec_str = line.strip().split(':')
    vec_str = vec_str[:-1].split(',')
    if len(vec_str) != vecdim:
        print('file_error')
        continue
    else:
        db_index_file.write("%s %s\n" % (str(db_index), face_ID))
        db_index += 1
        face_vec = [float(x) for x in vec_str]
        format_ = 'i' + 'f' * vecdim
        line_binary = struct.pack(format_, vecdim, *face_vec)      # len(bytes) = (1+128)*4=516
        db_bin_file.write(line_binary)
	if db_index % 1000 == 0:
            print "\r%d / %d" % (db_index, db_num),
db_bin_file.close()
db_index_file.close()
