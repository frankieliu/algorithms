import threading

class Semaphore(threading.Semaphore):
    wait = threading.Semaphore.acquire
    signal = threading.Semaphore.release

class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        self.start()
        