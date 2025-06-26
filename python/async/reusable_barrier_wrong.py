import inspect
from asyncio import Semaphore

def rendezvous(name):
    fn = inspect.currentframe().f_code.co_name
    print(f"{name}: {fn}")
 
def critical_point(name):
    fn = inspect.currentframe().f_code.co_name
    print(f"{name}: {fn}")

count = 0
n = 10
mutex = Semaphore(1)
turnstile = Semaphore(0)

rendezvous()

mutex.wait()
count += 1
if count == n:
    # after n it signals
    turnstile.signal()
mutex.signal()

# will NOT stuck here
# on second round because
# count goes back to n!

# problem:
# it is possible that single
# thread just keeps looping
# the problem will also
# show up any time not all
# threads go through count -=1

# perhaps it is better to
# decrement count before
# signalling the turnstile
turnstile.wait()
turnstile.signal()

critical_point()

mutex.wait()
count -= 1
# after passing through the
# turnstyle, then count
# decreases
if count == 0:
    turnstile.wait()
mutex.signal()