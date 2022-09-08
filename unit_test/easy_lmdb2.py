import lmdb
import pickle
import cv2
import numpy as np

image_dir= "./unit_test/train"
lmdb_path= "./unit_test/try"
digital_im_path= "./unit_test/0017.jpg"
env = lmdb.Environment(lmdb_path, subdir=True, map_size=1099511627776, readonly=False, # readonly=read only只读
                   meminit=False, map_async=True)    
txn = env.begin(write=True)

with open(digital_im_path.encode(), 'rb') as f:
    image = f.read()

txn.put(key = digital_im_path.encode(), value = image)
txn.commit()
print(env.stat()) 

############################################################
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
env.close()




