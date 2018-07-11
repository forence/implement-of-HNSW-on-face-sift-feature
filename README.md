# implement-of-HNSW-on-face-sift-feature
Use awesome HNSW method to generate an approximate K-nearest neighbor search which applied to extracted face sift featuree
## Face_data:
- face_sift_db.bin：保存database数据，数据格式为：[1（int）+128（float）]  *  db_num（411608）
- face_sift_query.bin：保存query数据，数据格式为：[1（int）+128（float）]  *  query_num（10000）
- face_sift_db/query_index.txt：index （从1 开始）+ photo_ID
- face_sift_gt.bin：保存每个query的groundtruth索引（top1000）,  数据格式为:  [1（int）+1000（int）]  *  query_num（10000）    
**Note:** int 和 float都是4字节
## 脚本：
- brute_force_gt.py：暴利搜索计算查询的groundtruth，输入face_sift_db/query.bin，输出生成文件格式：[1（int）+ top1000（int）] * query_num
- check_gt.py：查看某一查询对应的groundtruth，以检查真值计算是否正确
- data_merge.py：输入database.txt格式：photoID（str）+ feature（128 float），输出face_sift_db/query.bin，face_sift_db/query_index.txt    
**Note:** 本测试中，暴利搜索计算groundtruth采用的是计算向量之间的内积，所得结果越大说明向量之间距离越小，这是因为训练样本是经过归一化以后的结果（可以推倒内积和欧式距离在向量归一化的情况下成负相关）。原程序提供了3种度量距离的方法：
![](https://github.com/forence/implement-of-HNSW-on-face-sift-feature/blob/master/distance_method.png?raw=true)
