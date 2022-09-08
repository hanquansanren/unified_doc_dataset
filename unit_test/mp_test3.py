from multiprocessing import Pool
import time
def func(i): #返回值只有进程池才有,父子进程没有返回值
    time.sleep(1)
    return i*i

if __name__ == '__main__':
    p = Pool(5)
    ret = p.map(func,range(10))
    print(ret)