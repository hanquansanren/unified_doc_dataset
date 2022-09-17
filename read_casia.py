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
    # im=im[:,:,::-1]
    h=im.shape[0]*0.01
    w=im.shape[1]*0.01
    im = Image.fromarray(im)
    im.convert('RGB').save("./data_vis/img_{}.png".format(idx))
    # if lbl is not None:
    #     fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
    #     ax.imshow(im)
    #     ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
    #     ax.axis('off')
    #     plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
    #     # plt.tight_layout()
    #     plt.savefig('./synthesis_code/test/kk_{}.png'.format(idx))
    #     plt.close()


if __name__ == '__main__':
    env_db = lmdb.Environment('./ICDARTest_mdb_Alltlds_f2_py36_MSN_4')
    # env_db = lmdb.open("./test.lmdb")
    txn = env_db.begin()
    print(env_db.stat()) 


    for num, (key, value) in txn.cursor():  #遍历
        print(key)
        value=pickle.loads(value)
        check_vis(num, value[0][num][1], None)

    env_db.close()