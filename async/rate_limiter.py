import asyncio
import time
from collections import deque

class AsyncRateLimiter:
    def __init__(self, max_requests, time_window_seconds):
        self.max_requests = max_requests
        self.time_window = time_window_seconds

        # cannot have more that max_requests outstanding
        self.semaphore = asyncio.Semaphore(max_requests)
        
        # queue used to keep track of the outstanding requests inside the time_window
        self.request_times = deque(maxlen=max_requests)

        # internal lock for keeping track of deque and wait_time
        self.lock = asyncio.Lock()

    async def _wait_for_next_available(self):

        # Rate limiter works by making task wait
        # until thre 
        while True:  # Retry loop

            async with self.lock:
                now = time.time()

                # Remove expired requests
                while self.request_times and (now - self.request_times[0]) > self.time_window:
                    self.request_times.popleft()

                # you escape if all the requests are within the max_request
                if len(self.request_times) < self.max_requests:
                    return  # Exit when under limit

                # this is how long to wait until the oldest expires
                # and is removed from the window

                # Calculate wait time OUTSIDE the lock
                oldest = self.request_times[0]
                wait_time = self.time_window - (now - oldest)

            # Critical: sleep WITHOUT holding the lock
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def acquire(self):
        await self._wait_for_next_available()
        await self.semaphore.acquire()
        # add yourself to the dequeue
        # it contains only tasks that are running
        async with self.lock:
            self.request_times.append(time.time())

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.release()

import asyncio
import time
from collections import deque

class AsyncRateLimiter:
    def __init__(self, max_requests, time_window_seconds):
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.semaphore = asyncio.Semaphore(max_requests)
        
        # keep requests in a queue
        self.request_times = deque(maxlen=max_requests)

        # lock is not exposed
        self.lock = asyncio.Lock()

    async def _wait_for_next_available(self):

        async with self.lock:

            now = time.time()

            # 1. update the deque : only retain the requests within time_window
            # 2. 

            # Remove timestamps outside the current window
            # Remove requests_times outside of the windows
            # and corresponding semaphores
            while self.request_times and now - self.request_times[0] > self.time_window:
                self.request_times.popleft()
                self.semaphore.release()

            # if there are too many requests within the time window
            # look at the oldest request
            # and double the time to wait
            # then recheck!
            if len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                wait_time = self.time_window - (now - oldest)
                await asyncio.sleep(wait_time)
                await self._wait_for_next_available()  # Re-check after waiting

    async def acquire(self):
        # here is the blocking point
        await self._wait_for_next_available()
        await self.semaphore.acquire()

        async with self.lock:
            self.request_times.append(time.time())

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass  # No release needed for rate limiting

# Usage Example
async def worker(limiter, id):
    for i in range(5):
        async with limiter:
            print(f"Worker {id} started task {i} at {time.time():.2f}")
            await asyncio.sleep(0.3)  # Simulate work

async def main():
    limiter = AsyncRateLimiter(max_requests=3, time_window_seconds=1)
    tasks = [asyncio.create_task(worker(limiter, i)) for i in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main())