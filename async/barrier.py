# Barrier
# multiple threads are blocked
# only after $n$ arrive can the
# threads proceed

from asyncio import create_task, sleep, Semaphore
import asyncio
import tracemalloc
import numpy as np

mutex = Semaphore(1)
barrier = Semaphore(0)

""" Sleep random """
async def sr():
    await sleep(np.random.rand()*2)

counter = 1
n = 4 

async def a(name):

    global counter, n
    print(f"rendezvous {name}")
    await sr()

    await mutex.acquire()
    if counter == n:
        barrier.release()
    else:
        counter += 1
    mutex.release()
    
    await barrier.acquire()

    await mutex.acquire()
    if counter != 0:
        barrier.release()
    else:
        counter -= 1
    mutex.release()

    print(f" CR {name}")
    print(f"ending {name}")
    
async def main():
    tasks = [create_task(a(f"task {i}")) for i in range(n)]
    print("Tasks created")
    for task in tasks:
        await task
    print("Tasks ended")
    print(f"counter: {counter}")

tracemalloc.start()
asyncio.run(main(), debug=True)