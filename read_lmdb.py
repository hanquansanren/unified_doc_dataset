# -*- coding: utf-8 -*-
import lmdb
 
env_db = lmdb.Environment('test')
# env_db = lmdb.open("./trainC")
 
txn = env_db.begin()
 
# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
print txn.get(str(d1))
 
for key, value in txn.cursor():  #遍历
    print (key, value)
 
env_db.close()