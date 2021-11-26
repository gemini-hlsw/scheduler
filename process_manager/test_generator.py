import asyncio
from provider import TaskProvider
import signal


def shutdown(ptask, ctask):
    print("Shutting down")
    ptask.cancel()
    ctask.cancel()

async def main():
    provider = TaskProvider()
    queue = asyncio.Queue()
    ptask = asyncio.create_task(provider.producer(queue, 0.5))
    ctask = asyncio.create_task(provider.consumer(queue))
    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown, ptask, ctask)
    await ctask

if __name__ == '__main__':
   #sys.path.append("..")
    asyncio.run(main())
    
