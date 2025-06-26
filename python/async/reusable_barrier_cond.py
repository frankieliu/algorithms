# implementing reusable barrier with conditional variable
from asyncio import Condition,sleep,create_task,run
import random

cond1 = Condition()
cond2 = Condition()

t1 = 0
t2 = 0
n = 3
turns = 4
open_top = False
open_bot = False

async def sr():
    await sleep(random.random()*2)

async def a(name):
    global t1, t2, open_top, open_bot
    for i in range(turns):

        await sr()
        print(f"{name} {i}: rend")

        async with cond1:
            t1 += 1
            if t1 == n:
                t1 = 0
                open_top = True
                open_bot = False
                cond1.notify_all()
            await cond1.wait_for(lambda: open_top)

        await sr()
        print(f"{name} {i}: cr")

        async with cond2:
            t2 += 1
            if t2 == n:
                t2 = 0
                open_bot = True
                open_top = False
                cond2.notify_all()
            await cond2.wait_for(lambda: open_bot)

    print(f"{name} end")

async def main():
    tasks = [create_task(a(f"task {i}")) for i in range(n)]
    print("Created tasks")
    for task in tasks:
        await task 
    print("End tasks")

run(main())