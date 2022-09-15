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
    env = lmdb.Environment('./unit_test/train.lmdb')
    # env_db = lmdb.open("./test.lmdb")
    txn = env.begin()
    print(env.stat()) 


    # 查看数据
    txn=env.begin(write=False)
    for key, value in txn.cursor():
        key=key.decode()
        print(key)

    image_bin = txn.get(digital_im_path.encode())
    # 将二进制文件转为十进制文件（一维数组）
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    # 将数据转换(解码)成图像格式
    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
    img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    cv2.imwrite('./unit_test/show.jpg',img)
    ############################################################

    print("end")
