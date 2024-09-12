import asyncio
import pytest

from scheduler.services.visibility import visibility_calculator


@pytest.fixture(scope="session")
def visibility_calculator_fixture():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(visibility_calculator.calculate())