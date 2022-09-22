# coding=utf-8
# filename: merge_lmdb.py

import lmdb

# 将两个lmdb文件合并成一个新的lmdb
def merge_lmdb(lmdb1, lmdb2, result_lmdb):
    print ('Merge start!')
    env_1 = lmdb.open(lmdb1)
    env_2 = lmdb.open(lmdb2)
    txn_1 = env_1.begin(write=True)
    txn_2 = env_2.begin(write=True)
    txn_1.delete(b'__keys__')
    txn_1.delete(b'__len__')
    txn_2.delete(b'__keys__')
    txn_2.delete(b'__len__')
    txn_1.commit() 
    txn_2.commit()
    
    txn_1 = env_1.begin()
    txn_2 = env_2.begin()

    database_1 = txn_1.cursor()
    database_2 = txn_2.cursor()


    env_3 = lmdb.open(result_lmdb, map_size=int(1e12))
    txn_3 = env_3.begin(write=True)
    count = 0

    for (key, value) in database_1:
            txn_3.put((lmdb1[-6]+'_').encode()+key, value)
            count = count + 1
            if(count % 1000 == 0):
                txn_3.commit()
                count = 0
                txn_3 = env_3.begin(write=True)

    if(count % 1000 != 0):
        txn_3.commit()
        count = 0
        txn_3 = env_3.begin(write=True)

    for (key, value) in database_2:
        txn_3.put((lmdb2[-6]+'_').encode()+key, value)
        if(count % 1000 == 0):
            txn_3.commit()
            count = 0
            txn_3 = env_3.begin(write=True)

    if(count % 1000 != 0):
        txn_3.commit()
        count = 0
        txn_3 = env_3.begin(write=True)
    



    print (env_3.stat())
    env_1.close()
    env_2.close()
    env_3.close()

    print ('Merge success!')


def main():
    fr = open('./lmdb.txt')
    lmdb1 = fr.readline().strip() # warp_for_debug.lmdb
    lmdb2 = fr.readline().strip() # warp_for_debug2.lmdb
    output_lmdb = fr.readline().strip() # merged_lmdb.lmdb
    fr.close()

    merge_lmdb(lmdb1, lmdb2, output_lmdb)

if __name__ == '__main__':
  main()
