import asyncio
import signal
import logging
from queries import Session, new_schedule_mutation
from config import ODB_ENDPOINT_URL, CORE_ENDPOINT_URL
from aiorun import run


async def callback():
    # run query to set a new schedule
    s = Session(url=CORE_ENDPOINT_URL)
    try:
        a = await s.query(new_schedule_mutation)
        logging.info(f'{a=}')  # Respond from mutation
    except Exception as e:
        logging.info(e)  # fail to create new schedule


async def main():
    done = asyncio.Event()

    def shutdown():
        done.set()
        asyncio.get_event_loop().stop()

    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)
    s = Session(url=ODB_ENDPOINT_URL)
    print(ODB_ENDPOINT_URL)
    while not done.is_set():
        change = await s.subscribe_all()
        logging.debug(change)
        if change:
            await callback()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    try:
        run(main())
    except KeyboardInterrupt:
        pass
