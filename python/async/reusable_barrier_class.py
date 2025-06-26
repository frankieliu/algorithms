from asyncio import Semaphore

class Barrier():
    def __init__(self, n):
        self.ts1 = Semaphore(0)
        self.ts2 = Semaphore(0)
        self.mtx = Semaphore(1)
        self.n
        self.count = 0    

    async def phase1(self):
        async with self.mtx:
            self.count += 1
            if self.count == self.n:
                for i in range(self.n):
                    self.ts1.release()
        await self.ts1.acquire()

    async def phase2(self):
        async with self.mtx:
            self.count -= 1
            if self.count == 0:
                for i in range(self.n):
                    self.ts2.release()
        await self.ts2.acquire()

    def barrier(self):
        self.phase1()
        self.phase2()