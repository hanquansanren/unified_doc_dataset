from multiprocessing import Pool
import time
import numpy as np
# def ddd(a,b):
#     print("begin")
#     for ii in range(20):
#         for jj in range(20000):
#             print("gogogogogogo")
#             c=a*a+b*b
#     print("finish")
#     return c

# process_pool = Pool(5) # max=33

# d1 = process_pool.apply_async(func=ddd, args=(1.562,2.54556)).get()


# # for i in range(4):
# #     d1 = process_pool.apply_async(func=ddd, args=(1.562,2.54556)).get()
# #     d2 = process_pool.apply_async(func=ddd, args=(1.454,2.757)).get()
	
# process_pool.close()
# process_pool.join()



def run(fn):
    # fn: 函数参数是数据列表的一个元素
    print("begin")
    time.sleep(1)
    print(fn * fn)
    return (fn * fn)


if __name__ == "__main__":
    testFL = [1, 2, 3, 4, 5, 6]
    # print('shunxu:')  # 顺序执行(也就是串行执行，单进程)
    # s = time.time()
    # for fn in testFL:
    #     run(fn)
    # t1 = time.time()
    # print("顺序执行时间：", int(t1 - s))

    # print('concurrent:')  # 创建多个进程，并行执行
    # pool = Pool(3)  # 创建拥有3个进程数量的进程池
    # # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    # t1 = time.time()
    # pool.map(run, testFL)
    # # pool.map(run, testFL)
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出
    # t2 = time.time()
    # print("并行执行时间：", int(t2 - t1))



    print('concurrent2:')  # 创建多个进程，并行执行
    pool = Pool(8)  # 创建拥有3个进程数量的进程池
    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    t1 = time.time()
    for idx, fn in enumerate(testFL):
        aaa=pool.apply_async(run, (fn,)).get()
        bbb=pool.apply_async(run, (fn,)).get()
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    t2 = time.time()
    print("并行执行时间：", int(t2 - t1))

    # for idx1, type_path in enumerate(testFL):
    #         image_list = testFL
    #         process_pool = Pool(8) # max=33
    #         for idx2, image_path in enumerate(image_list):
    #             pickle_dict1 = process_pool.apply_async(func=run, args=(pjoin(data_dir, type_path, image_path),)).get()
    #             pickle_dict2 = process_pool.apply_async(func=run, args=(pjoin(data_dir, type_path, image_path),)).get()
    #             # # pickle_dict1 = get_syn_image(path=pjoin(data_dir, type_path, image_path), bg_path=bg_path, deform_type=deform_type1, idx='d1')
    #             # # pickle_dict2 = get_syn_image(path=pjoin(data_dir, type_path, image_path), bg_path=bg_path, deform_type=deform_type2, idx='d2')
    #             # txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'d1').encode(), value = pickle_dict1)
    #             # txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'d2').encode(), value = pickle_dict2)
    #             # txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'w1').encode(), value = w_dict)
    #             # txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'di').encode(), value = d_dict)
    #             # txn.commit()
    #             # txn = env.begin(write=True)

    #         process_pool.close()
    #         process_pool.join()