# Mutex
# thread a and b want to increment a counter

from asyncio import create_task, sleep, Semaphore
import asyncio
import tracemalloc
import numpy as np

lock = Semaphore(1)

""" Sleep random """
async def sr():
    await sleep(np.random.rand()*2)

counter = 0

async def a(name):
    global counter
    print(f"starting {name}")
    await sr()
    await lock.acquire()
    print(f"In CR {name}")
    counter += 1
    await sr() 
    print(f"Out CR {name}")
    lock.release()
    print(f"ending {name}")

async def main():
    tasks = [create_task(a("task a")), create_task(a("task b"))]
    print("Tasks created")
    for task in tasks:
        await task
    print("Tasks ended")
    print(f"counter: {counter}")

tracemalloc.start()
asyncio.run(main(), debug=True)