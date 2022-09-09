from multiprocessing import Pool
import time
kk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def fun(i): #返回值只有进程池才有,父子进程没有返回值
    print("begin")
    time.sleep(2)
    print("end")
    return i*i

if __name__ == '__main__':
    p = Pool(5)
    res_t = [] #从异步提交任务获取结果
    for i in range(10):
        res_l = []
        ret1 = p.apply_async(func=fun, args=(i,))
        ret2 = p.apply_async(func=fun, args=(i+10,))
        res_l.append(ret1)
        res_l.append(ret2)
        res_t.append(res_l[0].get())
        res_t.append(res_l[1].get())
    for j in res_t:  
        print(j) #等着func的计算结果
