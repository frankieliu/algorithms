# a1  b1
# a2  b2
#
# a1 and b1 must complete before a2 or b2 start

from asyncio import create_task, sleep, Semaphore
import asyncio
import tracemalloc
import numpy as np

lockb = Semaphore()
locka = Semaphore()

async def sr():
    await sleep(np.random.rand()*2)

async def a(name):
    print(f"starting {name}")
    await sr()
    lockb.release()
    await locka.acquire()
    print(f"ending {name}")

async def b(name):
    print(f"starting {name}")
    await sr()
    locka.release()
    await lockb.acquire()
    print(f"ending {name}")

async def main():
    tasks = [create_task(a("task a")), create_task(b("task b"))]
    print("Tasks created")
    for task in tasks:
        await task
    print("Tasks ended")

tracemalloc.start()
asyncio.run(main(), debug=True)