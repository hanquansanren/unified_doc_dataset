from multiprocessing import Pool
import time
def func(i): #返回值只有进程池才有,父子进程没有返回值
    time.sleep(1)
    return i*i

if __name__ == '__main__':
    p = Pool(5)
    res_l = [] #从异步提交任务获取结果
    for i in range(10):
        # res = p.apply(func,args=(i,)) #apply的结果就是func的返回值,同步提交
        # print(res)

        res = p.apply_async(func,args=(i,)) #apply_sync的结果就是异步获取func的返回值
        res_l.append(res) #从异步提交任务获取结果
    for res in res_l:  print(res.get()) #等着func的计算结果