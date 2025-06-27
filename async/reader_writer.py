"""
Simple explanation

Readers lock:
- Acquite: - wait until there are no writer present
  
  with writer_condition:
    while writer is present:
        writer_condition.await
    readers += 1 

- Release:
  decrement reader count if 0 then signal read_condition

Writer lock:
- Acquire: - announce there is a writer 

"""
import asyncio

class ReadersWriterLock:
    def __init__(self):
        # keep track of readers and writers
        self._readers = 0
        self._writer = False
        # used conditional variable
        self._read_condition = asyncio.Condition()
        self._write_condition = asyncio.Condition()

    async def acquire_read(self):
        async with self._write_condition:
            while self._writer:
                await self._write_condition.wait()
            self._readers += 1  # Now atomic with the writer check

    async def release_read(self):
        async with self._read_condition:
            self._readers -= 1
            if self._readers == 0:
                self._read_condition.notify_all()

    async def acquire_write(self):
        # writer can announce it is waiting
        async with self._write_condition:
            self._writer = True

        # writer waits here until there are no readers            
        async with self._read_condition:
            while self._readers > 0:
                await self._read_condition.wait()

    async def release_write(self):
        async with self._write_condition:
            self._writer = False
            self._write_condition.notify_all()

async def reader(lock, id, data):
    while True:
        await lock.acquire_read()
        try:
            print(f"Reader {id} reading data: {data}")
            await asyncio.sleep(0.1)  # Simulate reading work
        finally:
            await lock.release_read()
        await asyncio.sleep(0.5)  # Wait before reading again

async def writer(lock, data):
    counter = 0
    while True:
        # acquire the lock
        await lock.acquire_write()
        try:
            counter += 1
            data["value"] = counter
            print(f"Writer updating data to: {data}")
            await asyncio.sleep(0.3)  # Simulate writing work
        finally:
            await lock.release_write()
        await asyncio.sleep(1)  # Wait before writing again

async def main():
    # Sample data to be shared 
    data = {"value": 0}

    # Create a ReadersWriterLock
    lock = ReadersWriterLock()
    
    # Create multiple readers
    readers = [asyncio.create_task(reader(lock, i, data)) for i in range(5)]
    
    # Create one writer
    writer_task = asyncio.create_task(writer(lock, data))
    
    await asyncio.gather(*readers, writer_task)

asyncio.run(main())