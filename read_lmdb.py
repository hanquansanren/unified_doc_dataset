# -*- coding: utf-8 -*-
import pickle
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def check_vis(idx, im, lbl):
    '''
    im : distorted image   # HWC 
    lbl : fiducial_points  # 61*61*2 
    '''
    im=np.uint8(im)
    im=im[:,:,::-1]
    h=im.shape[0]*0.01
    w=im.shape[1]*0.01
    im = Image.fromarray(im)
    im.convert('RGB').save("./data_vis/img_{}.png".format(idx))
    
    # fig= plt.figure(j,figsize = (6,6))
    # fig, ax = plt.subplots(figsize = (10.24,7.68),facecolor='white')
    fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
    ax.imshow(im)
    ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
    ax.axis('off')
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
    # plt.tight_layout()
    plt.savefig('./synthesis_code/test/kk_{}.png'.format(idx))
    plt.close()


if __name__ == '__main__':
    # env_db = lmdb.Environment('./merged_lmdb_total_1.lmdb')
    # env_db = lmdb.Environment('./warp_for_debug.lmdb')
    env_db = lmdb.Environment('./warp3.lmdb')
    
    # env_db = lmdb.open("./test.lmdb")
    txn = env_db.begin()
    
    # get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
    # txn.get(str('d1').encode())
    print(env_db.stat()) 
    # with env_db.begin() as txn:
    #     with txn.cursor() as curs:
    #         print('key is:', curs.get('d1'.encode()))
    txn = env_db.begin()
    for key, value in txn.cursor():  #遍历
        print(key, type(value))
        # value=pickle.loads(value)
        # check_vis(8, value['image'], value['label'])

    env_db.close()