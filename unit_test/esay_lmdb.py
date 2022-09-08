import lmdb
 
env = lmdb.open("./simple", map_size=1099511627776)
txn = env.begin(write=True)
# 添加数据和键值
txn.put(key = '1'.encode(), value = 'aaa'.encode())
txn.put(key = '2'.encode(), value = 'bbb'.encode())
txn.put(key = '3'.encode(), value = 'ccc'.encode())
 





# 通过commit()函数提交更改
txn.commit()


############################################
# 查看数据
txn = env.begin()

# get函数通过键值查询数据
print(txn.get(str(2).encode()).decode())

# 通过cursor()遍历所有数据和键值
for key, value in txn.cursor():
    print(key, value)
############################################    


env.close()









# # 通过键值删除数据
# txn.delete(key = '1')
 
# # 修改数据
# txn.put(key = '3', value = 'ddd')