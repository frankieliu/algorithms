from asyncio import Semaphore,sleep,create_task,run
import random

mutex = Semaphore(1)
turnstile1 = Semaphore(0)
turnstile2 = Semaphore(0)

count = 0
n = 3
turns = 4

async def sr():
    await sleep(random.random()*2)

async def a(name):
    global count
    for i in range(turns):

        await sr()
        print(f"{name} {i}: rend")

        async with mutex:
            count += 1
            if count == n:
                for j in range(n):
                    turnstile1.release()
        
        await turnstile1.acquire()
        
        await sr() 
        print(f"{name} {i}: cr")

        async with mutex:
            count -= 1
            if count == 0:
                for j in range(n):
                    turnstile2.release()

        await turnstile2.acquire()

    print(f"{name} end")

async def main():
    tasks = [create_task(a(f"task {i}")) for i in range(n)]
    print("Created tasks")
    for task in tasks:
        await task 
    print("End tasks")

run(main())