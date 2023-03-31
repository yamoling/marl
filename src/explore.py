import os
import time
from multiprocessing import Process


def bg():
    # ignore first call 
    if os.fork() != 0:
        return
    print('sub process is running')
    time.sleep(5)
    print('sub process finished')


if __name__ == '__main__':
    p = Process(target=bg)
    p.start()
    p.join()
    print('exiting main')
    exit(0)
