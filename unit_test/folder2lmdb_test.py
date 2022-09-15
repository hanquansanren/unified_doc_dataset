import os
import os.path as osp
from os.path import join as pjoin

import lmdb
import pickle
import pickle
import cv2
import numpy as np


def image2byte(path):
    with open(path, 'rb') as f:
        bin_image = f.read()
    return bin_image


def folder2lmdb(data_path, mode="train"):
    data_dir = osp.expanduser(pjoin(data_path, mode)) # './unit_test/train'
    print("Loading dataset from {}".format(data_dir))
    
    type_list = os.listdir(pjoin(data_dir))
    lmdb_path = pjoin(data_path, "%s.lmdb" % mode) # './unit_test/train.lmdb'

    db = lmdb.open(lmdb_path, subdir=True,
                   map_size=1099511627776*2, readonly=False,
                   meminit=False, map_async=True)
    
    ###################################################
    '''写入数据'''
    txn = db.begin(write=True)
    id_sum=-1
    for idx1, type_path in enumerate(type_list):
        image_list = os.listdir(pjoin(data_dir, type_path))
        for idx2, image_path in enumerate(image_list):
            id_sum+=1
            # 这里只存了image，没有存label，如果需要存lable，需要自己写一个字典，分别设为value['image'], value['label']包装在一起，再写入38行的value参数
            txn.put(key='{0}_{1}_{2}'.format(idx1,image_path[0:4],id_sum).encode(), value=image2byte(pjoin(data_dir, type_path, image_path)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range((idx1+1)*(idx2+1))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    db.sync()

    ###################################################
    ''' 查看数据 '''
    print(db.stat()) 
    txn=db.begin(write=False)
    for key, value in txn.cursor():
        key=key.decode()
        print(key)
    # print(pickle.loads(txn.get(b'__len__')))
    for num, (key, value) in enumerate(txn.cursor()):
        if num<(pickle.loads(txn.get(b'__len__'))):
            image_bin = txn.get(key)
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
            cv2.imwrite('./unit_test/{}.jpg'.format(key.decode()), img)

    print("end")
    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default="./unit_test") 
    parser.add_argument('-s', '--split', type=str, default="train")
    args = parser.parse_args()

    folder2lmdb(args.folder, mode=args.split)
