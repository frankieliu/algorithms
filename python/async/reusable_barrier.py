# Reusable barrier
# after all the threads have gone through
# the barrier, want to reuse the barrier

# in previous implementation
# if counter == n: barrier.signal()
# doesn't work!
# 1. multiple threads may be behind
#    this line and all of them will
#    signal barrier
 
from asyncio import create_task, sleep, Semaphore
import asyncio
import tracemalloc
import numpy as np

mutex = Semaphore(1)
turnstile1 = Semaphore(0)
turnstile2 = Semaphore(1)

""" Sleep random """
async def sr():
    await sleep(np.random.rand()*2)

counter = 0
n = 3 


async def a(name, turns):
    global counter, n 

    for i in range(turns):

        # Rendezvous
        print(f"{name} {i}: rendez")
        await sr()

        await mutex.acquire()
        # print(f"{name} {i}: top {counter=}")
        counter += 1
        if counter == n:
            # lock the second
            await turnstile2.acquire()
            # open the first
            turnstile1.release()
        mutex.release()
        
        await turnstile1.acquire()
        turnstile1.release()

        await mutex.acquire()
        counter -= 1
        # print(f"{name} {i}: bot {counter=}") 
        if counter == 0:
            # lock the first
            await turnstile1.acquire()
            # open the second
            turnstile2.release()
        mutex.release()

        # Critical section
        print(f"{name} {i}: CR")
        await sr()

        await turnstile2.acquire()
        turnstile2.release()

    print(f"{name}: Ending")
    
async def main():
    turns = 4
    tasks = [create_task(a(f"task {i}", turns)) for i in range(n)]
    print("Tasks created")
    for task in tasks:
        await task
    print("Tasks ended")
    print(f"counter: {counter}")

tracemalloc.start()
asyncio.run(main(), debug=True)