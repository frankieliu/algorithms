import asyncio
import time
from collections import deque
from typing import Optional

class AsyncRateLimiter:
    def __init__(self, max_requests: int, time_window_seconds: float):
        if max_requests <= 0 or time_window_seconds <= 0:
            raise ValueError("max_requests and time_window_seconds must be positive")
        
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.semaphore = asyncio.Semaphore(max_requests)
        self.lock = asyncio.Lock()
        self.request_times = deque(maxlen=max_requests)  # Tracks start times of active tasks
        self._cleanup_task: Optional[asyncio.Task] = None

    async def _cleanup_expired(self):
        """Background task to cleanup expired requests"""
        while True:
            async with self.lock:
                now = time.time()
                while self.request_times and (now - self.request_times[0]) > self.time_window:
                    self.request_times.popleft()
                    self.semaphore.release()
            
            await asyncio.sleep(self.time_window / 2)  # Cleanup frequency

    async def __aenter__(self):
        # Start cleanup task on first use
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        
        await self.semaphore.acquire()  # Block until permit available

        async with self.lock:
            if len(self.request_times) >= self.max_requests:
                # Handle case where semaphore was acquired but window is full (shouldn't happen)
                oldest = self.request_times[0]
                wait_time = max(0.0, self.time_window - (time.time() - oldest))
                self.semaphore.release()
                await asyncio.sleep(wait_time)
                return await self.__aenter__()  # Retry
            
            self.request_times.append(time.time())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass  # Semaphore released by cleanup task

    async def close(self):
        """Cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
async def worker(limiter, id):
    async with limiter:
        print(f"Worker {id} started at {time.time():.2f}")
        await asyncio.sleep(0.8)  # Simulate work

async def main():
    limiter = AsyncRateLimiter(max_requests=2, time_window_seconds=1)
    try:
        tasks = [asyncio.create_task(worker(limiter, i)) for i in range(5)]
        await asyncio.gather(*tasks)
    finally:
        await limiter.close()

asyncio.run(main())